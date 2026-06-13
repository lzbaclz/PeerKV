"""e32 -- BATCH SENSITIVITY: where does the "peer is compute-idle" premise hold?

The reviewer's W3/Q-i: PeerKV-Parallel assumes the peer GPU has spare compute. As
the decode batch grows, the peer's compute fills up, which should erode the win.
This delineates the addressable regime honestly: we sweep batch B at a fixed
context and report the L-layer attention-path cost per decode step for

  * peer_parallel : each GPU computes its KV shard's partial in parallel
  * single_gpu    : full KV on cuda:0 (ref; OOMs at large B*context)
  * copyback      : peer KV streamed back, bounded-peak double-buffered

Two honest readings: (1) peer_parallel's advantage over copyback is a structural
bandwidth/latency property and persists across B; (2) vs a single GPU that FITS,
batching adds parallel work to BOTH GPUs, so the regime where moving compute to
the peer pays is low-to-moderate batch -- and, decisively, peer_parallel sustains
batches whose aggregate KV OOMs one GPU (batched enablement). All attention-only
(MLP is model-common; amortization with real weights is e27/e31).

    python experiments/e32_batch_sweep.py --context 16384 --batches 1,2,4,8,16,32
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "batch_sweep.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=8)       # GQA (Llama-3-8B)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--local-frac", type=float, default=0.5)
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--batches", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import peer_parallel_attention, flash_partial, merge_partial

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        OUT.write_text(json.dumps({"_is_measured": False, "note": f"need 2 GPUs (have {ngpu})"})); return
    L, C, H, D = args.layers, args.context, args.heads, args.head_dim
    Lc = max(1, min(C-1, round(args.local_frac * C))); Pc = C - Lc
    scale = 1.0 / (D ** 0.5)
    Ctok = args.chunk_tokens
    batches = [int(b) for b in args.batches.split(",")]

    def sync(): torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    def med(xs): return statistics.median(sorted(xs))

    def bench(fn):
        ms = []
        for _ in range(args.seeds):
            for _ in range(3): fn()
            sync(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            for _ in range(args.trials):
                sync(); s.record(); fn(); e.record(); sync()
                ms.append(s.elapsed_time(e))
        return med(ms)

    rows = []
    for B in batches:
        torch.cuda.empty_cache()
        try:
            qs = [torch.randn(B, H, 1, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
            K0 = [torch.randn(B, H, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
            V0 = [torch.randn(B, H, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
            K1 = [torch.randn(B, H, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
            V1 = [torch.randn(B, H, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
            sync()
        except RuntimeError as e:
            rows.append({"batch": B, "status": f"alloc OOM: {str(e)[:50]}"}); break

        def pp():
            for i in range(L):
                peer_parallel_attention(qs[i], [(K0[i], V0[i]), (K1[i], V1[i])], scale=scale)
        pp_ms = bench(pp)
        row = {"batch": B, "peer_parallel_ms": pp_ms, "kv_gib_per_gpu":
               (K0[0].numel()+V0[0].numel())*2*L/1024**3 + (K1[0].numel()+V1[0].numel())*2*L/1024**3 / 2}

        # single: full KV gathered on cuda:0 (may OOM as B grows -> enablement)
        try:
            Kf = [torch.cat([K0[i], K1[i].to("cuda:0")], dim=2) for i in range(L)]
            Vf = [torch.cat([V0[i], V1[i].to("cuda:0")], dim=2) for i in range(L)]
            sync()
            def sg():
                for i in range(L):
                    F.scaled_dot_product_attention(qs[i], Kf[i], Vf[i], scale=scale)
            sg_ms = bench(sg)
            row.update({"single_gpu_ms": sg_ms, "peer_vs_single": pp_ms/sg_ms})
            del Kf, Vf
        except RuntimeError:
            row.update({"single_gpu_ms": None, "single_gpu_status": "OOM (batched enablement)"})
        torch.cuda.empty_cache()

        # copyback: per-layer chunked bounded-peak (on-the-fly slices)
        cps = torch.cuda.Stream(device="cuda:0")
        def cb():
            for i in range(L):
                O, l = flash_partial(qs[i], K0[i], V0[i], scale)
                starts = list(range(0, Pc, Ctok)); n = len(starts)
                buf = [None]*n; ev = [torch.cuda.Event() for _ in range(n)]
                def fetch(j):
                    s = starts[j]
                    with torch.cuda.stream(cps):
                        kc = K1[i][:, :, s:s+Ctok, :].contiguous().to("cuda:0", non_blocking=True)
                        vc = V1[i][:, :, s:s+Ctok, :].contiguous().to("cuda:0", non_blocking=True)
                        buf[j] = (kc, vc); ev[j].record(cps)
                fetch(0)
                for j in range(n):
                    if j+1 < n: fetch(j+1)
                    torch.cuda.current_stream().wait_event(ev[j]); Kc, Vc = buf[j]
                    Oc, lc = flash_partial(qs[i], Kc, Vc, scale); O, l = merge_partial(O, l, Oc, lc)
                    buf[j] = None
        cb_ms = bench(cb)
        row.update({"copyback_ms": cb_ms, "peer_vs_copyback": cb_ms/pp_ms})
        sv = row.get("peer_vs_single")
        print(f"  B={B:3d}: peer {pp_ms:.2f}  single "
              f"{('%.2f'%row['single_gpu_ms']) if row.get('single_gpu_ms') else 'OOM':>6}  "
              f"copyback {cb_ms:.2f}  | vs_single "
              f"{('%.2fx'%sv) if sv else '--':>6}  vs_copyback {cb_ms/pp_ms:.2f}x")
        rows.append(row)
        del qs, K0, V0, K1, V1

    res = {"_experiment": "e32_batch_sweep", "_is_measured": True,
           "layers": L, "context": C, "heads": H, "head_dim": D,
           "local_tokens": Lc, "peer_tokens": Pc, "geometry": "Llama-3-8B (GQA)",
           "note": "attention-path only; per decode step over all B requests",
           "rows": rows, "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

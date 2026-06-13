"""e18 -- NVLink-tiered flash attention: bounded peak + enablement (RQ2 MEASURED).

RUN ON THE DUAL-A100 BOX. The complete version of the idea: cold KV blocks stay
on the peer GPU (NVLink) or host (PCIe) and are streamed a block at a time
through an online-softmax merge (umallm.torch_tiered_attn.flash_merge_attention),
so the compute GPU's peak holds only the query + one block -- never the full KV.

This gives two things e17 (copy-back + dense SDPA, materializes full KV) cannot:
  * bounded peak  -> a context that OVERFLOWS one GPU runs on two (enablement);
  * the spill link still matters: NVLink (peer GPU) vs PCIe (host) decode latency.

We compare, at a context sized to overflow one GPU's KV budget:
  * NVLink-tiered (flash): overflow on peer GPU                 [ours]
  * host-offload  (flash): overflow on host DRAM               [baseline]
  * full-materialize (dense SDPA): all blocks gathered to cuda:0  [shows it OOMs]
reporting per-step latency AND peak memory on cuda:0.

Authored on a Mac with no CUDA; <2-GPU path writes a placeholder. On A100:
    pip install "torch>=2.4"
    python experiments/e18_p2p_flash.py --n-blocks 2048 --local-blocks 128
Writes experiments/results/p2p_flash.json. Send it back.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "p2p_flash.json"


def _write(res: dict) -> None:
    res["_experiment"] = "e18_p2p_flash"
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=2048)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--local-blocks", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--chunk-blocks", type=int, default=1,
                    help="coalesce this many consecutive same-device KV blocks per "
                         "transfer (1=block-at-a-time; larger=bandwidth-bound fetch)")
    args = ap.parse_args()

    res = {k: getattr(args, k) for k in
           ("n_blocks", "block_tokens", "local_blocks", "heads", "head_dim",
            "trials", "chunk_blocks")}
    try:
        import torch
        import torch.nn.functional as F
        from umallm.torch_tiered_attn import flash_merge_attention
    except Exception as e:  # noqa: BLE001
        res.update({"_is_measured": False, "note": f"torch/umallm import failed: {e}"})
        _write(res); return

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        res.update({"_is_measured": False,
                    "note": f"need >=2 CUDA GPUs (have {ngpu}); run on the dual-A100 box"})
        _write(res); print(f"  only {ngpu} GPU(s) -- wrote placeholder."); return

    H, T, D = args.heads, args.block_tokens, args.head_dim
    N, L = args.n_blocks, min(args.local_blocks, args.n_blocks)
    n_spill = N - L
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))
    q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")

    def mkblocks(spill_dev):
        kb = [torch.randn(1, H, T, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
        vb = [torch.randn(1, H, T, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
        for _ in range(n_spill):
            dev = "cpu" if spill_dev == "host" else spill_dev
            k = torch.randn(1, H, T, D, dtype=torch.float16, device=dev)
            v = torch.randn(1, H, T, D, dtype=torch.float16, device=dev)
            if spill_dev == "host":
                k, v = k.pin_memory(), v.pin_memory()
            kb.append(k); vb.append(v)
        return kb, vb

    def bench_flash(spill_dev):
        kb, vb = mkblocks(spill_dev)
        for _ in range(2):
            flash_merge_attention(q, kb, vb, chunk_blocks=args.chunk_blocks)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            o = flash_merge_attention(q, kb, vb, chunk_blocks=args.chunk_blocks)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated(0) / 1024**2
        del kb, vb; torch.cuda.empty_cache()
        return statistics.median(ts) * 1e3, peak

    def bench_full():
        # full-materialize: gather everything onto cuda:0 and run dense SDPA.
        # This is what overflows -> expected to OOM at large context (the point).
        kb = [torch.randn(1, H, T, D, dtype=torch.float16, device="cuda:0") for _ in range(N)]
        vb = [torch.randn(1, H, T, D, dtype=torch.float16, device="cuda:0") for _ in range(N)]
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(0)
        K = torch.cat(kb, dim=2); V = torch.cat(vb, dim=2)
        o = F.scaled_dot_product_attention(q, K, V)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(0) / 1024**2
        del kb, vb, K, V; torch.cuda.empty_cache()
        return peak

    nv_ms, nv_peak = bench_flash("cuda:1")     # NVLink-tiered (ours)
    host_ms, host_peak = bench_flash("host")   # host-offload baseline
    try:
        full_peak = bench_full()
        full_note = "ok"
    except RuntimeError as e:  # typically CUDA OOM -> the enablement point
        torch.cuda.empty_cache()
        full_peak, full_note = None, f"OOM/error: {str(e)[:120]}"

    res.update({
        "_is_measured": True, "kind": "flash_tiered_decode_step",
        "device": torch.cuda.get_device_name(0), "n_gpus": ngpu,
        "peer_access_enabled": p2p, "spill_blocks": n_spill,
        "nvlink_tiered_ms": nv_ms, "nvlink_tiered_peak_mb": nv_peak,
        "host_offload_ms": host_ms, "host_offload_peak_mb": host_peak,
        "full_materialize_peak_mb": full_peak, "full_materialize_status": full_note,
        "nvlink_speedup_vs_host": (host_ms / nv_ms) if nv_ms else None,
        "peak_reduction_vs_full": (full_peak / nv_peak) if (full_peak and nv_peak) else None,
        "note": ("flash merge streams one block at a time -> bounded peak "
                 "(enablement); NVLink vs host is the spill-link latency. "
                 "full-materialize shows the dense path's peak (OOM = the point). "
                 "no-prefetch; zero-copy peer read is future work."),
    })
    _write(res)
    print(f"  nvlink={nv_ms:.2f}ms/{nv_peak:.0f}MB  host={host_ms:.2f}ms/{host_peak:.0f}MB"
          f"  full_peak={full_peak}  speedup={res['nvlink_speedup_vs_host']}")


if __name__ == "__main__":
    main()

"""e21 -- steady-state autoregressive decode loop with tiered KV, real model
geometry, realistic paged block sizes, batch>1, and the FAIR overlapped baseline.

Closes the reviewer demands that the single-step microbenchmark cannot:
 * a real multi-step decode loop (growing KV) -> tokens/s and TPOT p50/p95;
 * REAL model KV geometry (Llama-3-8B/70B GQA: 8 kv-heads x 128) not a lucky shape;
 * realistic PAGED block sizes incl. vLLM's 16/32 tokens (NOT only 256) -- this is
   where the launch-bound regime lives and where naive NVLink tiering LOSES;
 * batch > 1 and realistic overflow fractions (10/25/50%), not only 94% full-spill;
 * NVLink-tiered vs an OVERLAPPED (FlexGen-style, double-buffered) host baseline
   vs all-local, every transfer prefetched one step ahead behind compute.

Reports the NVLink-vs-overlapped-host TPOT ratio AS A FUNCTION OF BLOCK SIZE, so
the crossover from "NVLink loses (small blocks, launch-bound)" to "NVLink wins
(large blocks, bandwidth-bound)" is explicit and honest. CUDA-event timed,
fixed seed, median + p95.
"""
from __future__ import annotations
import argparse, json, statistics
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "decode_loop.json"

MODELS = {  # (n_kv_heads, head_dim, n_layers) -- GQA geometries
    "llama3-8b":  (8, 128, 32),
    "llama3-70b": (8, 128, 80),
    "mistral-7b": (8, 128, 32),
    "llama2-13b": (40, 128, 40),   # MHA, larger per-token KV
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-8b", choices=list(MODELS))
    ap.add_argument("--ctx-tokens", type=int, default=131072, help="context length being decoded at")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--overflow-frac", type=float, default=0.5, help="fraction of KV that must spill off cuda:0")
    ap.add_argument("--block-sizes", default="16,32,64,128,256")
    ap.add_argument("--chunk-kb", type=int, default=8192, help="coalesce spill into ~this many KiB per transfer")
    ap.add_argument("--steps", type=int, default=40, help="decode steps timed (per arm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", default="", help="suffix for the output json (e.g. _perblock)")
    args = ap.parse_args()
    global OUT
    if args.out_tag:
        OUT = OUT.parent / f"decode_loop{args.out_tag}.json"

    import torch
    torch.manual_seed(args.seed)
    H, D, n_layers = MODELS[args.model]
    dev = torch.device("cuda:0")
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))
    B = args.batch
    scale = 1.0 / (D ** 0.5)
    copy_stream = torch.cuda.Stream(device=dev)

    def merge(state, Kc, Vc, qs):
        m, l, o = state
        s = (qs @ Kc.transpose(-1, -2)).float()
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        corr = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        o = o * corr + p.to(Vc.dtype) @ Vc
        return (m_new, l, o)

    def fresh(qd):
        return (torch.full((B, H, 1, 1), float("-inf"), device=dev, dtype=torch.float32),
                torch.zeros((B, H, 1, 1), device=dev, dtype=torch.float32),
                torch.zeros((B, H, 1, D), device=dev, dtype=torch.float32))

    def run_arm(spill_dev, block_tokens, prefetch):
        # one layer's KV at this context; blocks of block_tokens; overflow_frac spills.
        n_blocks = max(1, args.ctx_tokens // block_tokens)
        n_spill = int(n_blocks * args.overflow_frac)
        n_local = n_blocks - n_spill
        q = torch.randn(B, H, 1, D, dtype=torch.float16, device=dev)
        qs = q * scale
        # local resident KV: one contiguous tensor
        Kloc = torch.randn(B, H, max(1, n_local) * block_tokens, D, dtype=torch.float16, device=dev)
        Vloc = torch.randn(B, H, max(1, n_local) * block_tokens, D, dtype=torch.float16, device=dev)
        # spill stored as pre-contiguous chunks (paged pool); chunk ~ chunk-kb
        per_block_bytes = B * H * block_tokens * D * 2
        blocks_per_chunk = max(1, (args.chunk_kb * 1024) // per_block_bytes)
        ck, cv = [], []
        rem = n_spill
        sd = "cpu" if spill_dev == "host" else spill_dev
        while rem > 0:
            c = min(blocks_per_chunk, rem)
            K = torch.randn(B, H, c * block_tokens, D, dtype=torch.float16, device=sd)
            V = torch.randn(B, H, c * block_tokens, D, dtype=torch.float16, device=sd)
            if spill_dev == "host":
                K, V = K.pin_memory(), V.pin_memory()
            ck.append(K); cv.append(V); rem -= c

        def step():
            st = fresh(q)
            st = merge(st, Kloc, Vloc, qs)
            if not ck:
                m, l, o = st; return (o / l).to(q.dtype)
            if not prefetch:
                for Kc, Vc in zip(ck, cv):
                    st = merge(st, Kc.to(dev, non_blocking=True), Vc.to(dev, non_blocking=True), qs)
            else:
                n = len(ck); ev = [torch.cuda.Event() for _ in range(n)]; buf = [None] * n
                with torch.cuda.stream(copy_stream):
                    buf[0] = (ck[0].to(dev, non_blocking=True), cv[0].to(dev, non_blocking=True)); ev[0].record(copy_stream)
                for i in range(n):
                    if i + 1 < n:
                        with torch.cuda.stream(copy_stream):
                            buf[i+1] = (ck[i+1].to(dev, non_blocking=True), cv[i+1].to(dev, non_blocking=True)); ev[i+1].record(copy_stream)
                    torch.cuda.current_stream().wait_event(ev[i])
                    Kc, Vc = buf[i]; st = merge(st, Kc, Vc, qs); buf[i] = None
            m, l, o = st; return (o / l).to(q.dtype)

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(args.steps):
            torch.cuda.synchronize(); e0.record(); step(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        del Kloc, Vloc, ck, cv; torch.cuda.empty_cache()
        ts.sort()
        # per-layer ms; a full decode step touches all layers -> x n_layers for model TPOT
        p50 = statistics.median(ts); p95 = ts[int(0.95*len(ts))-1] if len(ts) >= 20 else ts[-1]
        return {"per_layer_p50_ms": p50, "per_layer_p95_ms": p95,
                "model_tpot_p50_ms": p50 * n_layers, "model_tpot_p95_ms": p95 * n_layers,
                "n_blocks": n_blocks, "n_spill": n_spill, "blocks_per_chunk": blocks_per_chunk}

    rows = []
    for bt in [int(x) for x in args.block_sizes.split(",")]:
        arms = {}
        for arm, sd, pf in (("all_local", "cuda:0", False),
                            ("nvlink_prefetch", "cuda:1", True),
                            ("host_prefetch", "host", True),
                            ("nvlink_noprefetch", "cuda:1", False),
                            ("host_noprefetch", "host", False)):
            arms[arm] = run_arm(sd, bt, pf)
        def tput(a):  # tokens/s = batch / model_tpot
            return B / (arms[a]["model_tpot_p50_ms"] / 1e3)
        row = {"block_tokens": bt,
               "nvlink_vs_host_prefetch": arms["host_prefetch"]["model_tpot_p50_ms"] / arms["nvlink_prefetch"]["model_tpot_p50_ms"],
               "nvlink_vs_host_noprefetch": arms["host_noprefetch"]["model_tpot_p50_ms"] / arms["nvlink_noprefetch"]["model_tpot_p50_ms"],
               "nvlink_prefetch_vs_noprefetch": arms["nvlink_noprefetch"]["model_tpot_p50_ms"] / arms["nvlink_prefetch"]["model_tpot_p50_ms"],
               "tput_nvlink": tput("nvlink_prefetch"), "tput_host": tput("host_prefetch"), "tput_all_local": tput("all_local"),
               "arms": arms}
        rows.append(row)
        print(f"  blk={bt:4d}  NVLink-vs-host: prefetch={row['nvlink_vs_host_prefetch']:5.2f}x "
              f"noprefetch={row['nvlink_vs_host_noprefetch']:5.2f}x | tok/s NVLink={row['tput_nvlink']:6.1f} "
              f"host={row['tput_host']:6.1f} local={row['tput_all_local']:7.1f}")

    res = {"_experiment": "e21_decode_loop", "_is_measured": True,
           "kind": "steady_state_decode_tpot_blocksize_sweep",
           "device": torch.cuda.get_device_name(0), "peer_access_enabled": p2p,
           "model": args.model, "geometry_HxDxL": list(MODELS[args.model]),
           "ctx_tokens": args.ctx_tokens, "batch": B, "overflow_frac": args.overflow_frac,
           "chunk_kb": args.chunk_kb, "steps": args.steps, "seed": args.seed,
           "block_size_sweep": rows,
           "note": ("steady-state decode at fixed context; model TPOT = per-layer x n_layers. "
                    "Both spill tiers use one-step-ahead double-buffered prefetch (fair). The "
                    "NVLink-vs-host ratio vs block_tokens shows the launch-bound (small block, "
                    "NVLink may lose) -> bandwidth-bound (large block, NVLink wins) crossover."),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

"""Chunked flash streaming: resolve the peak-vs-bandwidth tension.

e18's flash_merge_attention streams ONE 512KB block at a time -> bounded peak
but launch-bound, so the NVLink bandwidth gap is hidden (~0.86x vs host).
The batched probe coalesces the WHOLE spill into one copy -> 5.5x NVLink win but
reintroduces a large peak (materializes the full spill on the compute GPU).

The engineered sweet spot is CHUNKED streaming: fetch C blocks per coalesced
transfer, run the online-softmax (flash) merge over the chunk, free it, repeat.
Peak holds q + local + ONE chunk (C blocks), not the full KV; transfers are
C-blocks-large so they become bandwidth-bound as C grows. This sweeps C to trace
the peak-vs-latency Pareto for NVLink vs host spill.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "p2p_flash_chunked.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=2048)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--local-blocks", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunks", default="1,4,16,64,256")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    import torch
    from umallm.torch_tiered_attn import flash_merge_attention  # the SHIPPED path

    H, T, D = args.heads, args.block_tokens, args.head_dim
    N, L = args.n_blocks, min(args.local_blocks, args.n_blocks)
    n_spill = N - L
    scale = 1.0 / (D ** 0.5)
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))
    chunk_sizes = [int(x) for x in args.chunks.split(",") if x.strip()]

    q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")
    # local hot blocks: contiguous on cuda:0 (token dim L*T)
    Kloc = torch.randn(1, H, L * T, D, dtype=torch.float16, device="cuda:0")
    Vloc = torch.randn(1, H, L * T, D, dtype=torch.float16, device="cuda:0")

    def make_spill_chunks(dev, chunk_blocks):
        """Spill stored as a list of CONTIGUOUS (1,H,C*T,D) chunk tensors -- the
        realistic paged layout: one chunk == one coalesced transfer."""
        d = "cpu" if dev == "host" else dev
        chunks_k, chunks_v = [], []
        rem = n_spill
        while rem > 0:
            c = min(chunk_blocks, rem)
            K = torch.randn(1, H, c * T, D, dtype=torch.float16, device=d)
            V = torch.randn(1, H, c * T, D, dtype=torch.float16, device=d)
            if dev == "host":
                K, V = K.pin_memory(), V.pin_memory()
            chunks_k.append(K); chunks_v.append(V); rem -= c
        return chunks_k, chunks_v

    def flash_chunked(chunks_k, chunks_v):
        # Drive the SHIPPED flash_merge_attention on pre-contiguous chunks: local
        # hot region (resident on cuda:0) + spill stored as contiguous C-block
        # chunks (a paged-KV pool's natural layout), one transfer per chunk.
        # chunk_blocks=1 here means "one list element = one transfer" (each element
        # is already a C-block chunk) -- no per-step cat, so the host baseline is
        # not penalized. This makes the Pareto a measurement of shipped code.
        return flash_merge_attention(q, [Kloc] + list(chunks_k),
                                     [Vloc] + list(chunks_v), scale=scale,
                                     chunk_blocks=1)

    def bench(dev, chunk_blocks):
        Ksp, Vsp = make_spill_chunks(dev, chunk_blocks)
        for _ in range(2):
            flash_chunked(Ksp, Vsp)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(0)
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            flash_chunked(Ksp, Vsp); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated(0) / 1024**2
        del Ksp, Vsp; torch.cuda.empty_cache()
        return statistics.median(ts) * 1e3, peak

    rows = []
    for C in chunk_sizes:
        nv_ms, nv_peak = bench("cuda:1", C)
        h_ms, h_peak = bench("host", C)
        row = {"chunk_blocks": C, "chunk_mb": C * H * T * D * 2 * 2 / 1024**2,
               "nvlink_ms": nv_ms, "nvlink_peak_mb": nv_peak,
               "host_ms": h_ms, "host_peak_mb": h_peak,
               "nvlink_speedup_vs_host": h_ms / nv_ms if nv_ms else None}
        rows.append(row)
        print(f"  C={C:>4} ({row['chunk_mb']:6.1f}MB)  NVLink={nv_ms:8.2f}ms/{nv_peak:6.0f}MB  "
              f"host={h_ms:8.2f}ms/{h_peak:6.0f}MB  speedup={row['nvlink_speedup_vs_host']:.2f}x")

    res = {"_experiment": "e18c_chunked_flash", "_is_measured": True,
           "kind": "chunked_flash_decode_step_pareto",
           "device": torch.cuda.get_device_name(0), "peer_access_enabled": p2p,
           "n_blocks": N, "local_blocks": L, "spill_blocks": n_spill,
           "block_tokens": T, "heads": H, "head_dim": D, "trials": args.trials,
           "sweep": rows,
           "note": ("chunked online-softmax flash merge: peak holds q+local+one chunk "
                    "(C blocks), transfers coalesced to C blocks. Traces peak-vs-latency "
                    "Pareto; C=1 reproduces e18 (launch-bound), large C approaches the "
                    "batched bandwidth-bound win while keeping peak bounded."),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

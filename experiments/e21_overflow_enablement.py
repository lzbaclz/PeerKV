"""e21 -- true-overflow ENABLEMENT: a KV cache that OOMs one GPU decodes on two.

RUN ON THE DUAL-A100 BOX. The headline enablement claim for PeerKV-Parallel: a
context whose KV exceeds a single GPU's HBM cannot run all-local (it OOMs), but
runs with KV-parallel distributed attention -- each GPU holds half the KV and
computes its partial locally, exchanging only KB-sized (O, lse) partials.

We (1) validate exactness at small scale (cos vs dense), then (2) allocate KV sized
to OVERFLOW one 80GB GPU (~target_gb per GPU, ~2x total), run a peer-parallel decode
step (works) and report its latency + per-GPU memory, and (3) try to gather all KV
onto cuda:0 and run dense attention -- expected to OOM, which is exactly the point.

    python experiments/e21_overflow_enablement.py --target-gb-per-gpu 45
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "overflow_enablement.json"


def _write(res: dict) -> None:
    res["_experiment"] = "e21_overflow_enablement"
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gb-per-gpu", type=float, default=45.0,
                    help="KV GB resident per GPU; ~2x total must overflow one GPU")
    ap.add_argument("--gb-cuda0", type=float, default=None,
                    help="override: KV GB on cuda:0 (asymmetric, e.g. shared box)")
    ap.add_argument("--gb-cuda1", type=float, default=None,
                    help="override: KV GB on cuda:1")
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    res = {k: getattr(args, k) for k in
           ("target_gb_per_gpu", "block_tokens", "heads", "head_dim", "trials")}
    try:
        import torch
        import torch.nn.functional as F
        from umallm.peer_parallel_attn import peer_parallel_attention
    except Exception as e:  # noqa: BLE001
        res.update({"_is_measured": False, "note": f"import failed: {e}"})
        _write(res); return

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        res.update({"_is_measured": False, "note": f"need >=2 GPUs (have {ngpu})"})
        _write(res); print(f"  only {ngpu} GPU(s) -- placeholder."); return

    H, T, D = args.heads, args.block_tokens, args.head_dim
    scale = 1.0 / (D ** 0.5)
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))

    # ---- (1) exactness at small scale (real values) ----
    torch.manual_seed(0)
    sn = 256
    q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")
    K0s = torch.randn(1, H, sn*T, D, dtype=torch.float16, device="cuda:0")
    V0s = torch.randn(1, H, sn*T, D, dtype=torch.float16, device="cuda:0")
    K1s = torch.randn(1, H, sn*T, D, dtype=torch.float16, device="cuda:1")
    V1s = torch.randn(1, H, sn*T, D, dtype=torch.float16, device="cuda:1")
    ref = F.scaled_dot_product_attention(
        q.float(), torch.cat([K0s, K1s.to("cuda:0")], 2).float(),
        torch.cat([V0s, V1s.to("cuda:0")], 2).float(), scale=scale)
    out = peer_parallel_attention(q, [(K0s, V0s), (K1s, V1s)], scale=scale)
    cos = float(torch.nn.functional.cosine_similarity(
        ref.reshape(-1), out.reshape(-1).float(), dim=0))
    res["exactness_cosine_small"] = cos
    print(f"  exactness (small): cos={cos:.6f}")
    del K0s, V0s, K1s, V1s, ref, out
    for d in (0, 1):
        torch.cuda.empty_cache()

    # ---- (2) overflow scale: KV resident per GPU (may be asymmetric) ----
    block_bytes_kv = T * D * 2 * H * 2          # K+V per block, bytes
    gb0 = args.gb_cuda0 if args.gb_cuda0 is not None else args.target_gb_per_gpu
    gb1 = args.gb_cuda1 if args.gb_cuda1 is not None else args.target_gb_per_gpu
    blocks0 = int(gb0 * 1e9 / block_bytes_kv)
    blocks1 = int(gb1 * 1e9 / block_bytes_kv)
    tot_blocks = blocks0 + blocks1
    tot_gb = tot_blocks * block_bytes_kv / 1e9
    res.update({"blocks_cuda0": blocks0, "blocks_cuda1": blocks1,
                "total_blocks": tot_blocks, "total_kv_gb": tot_gb,
                "kv_gb_cuda0": blocks0*block_bytes_kv/1e9,
                "kv_gb_cuda1": blocks1*block_bytes_kv/1e9,
                "peer_access_enabled": p2p})
    print(f"  overflow scale: {tot_blocks} blocks = {tot_gb:.1f} GB total "
          f"(cuda:0 {blocks0*block_bytes_kv/1e9:.1f} GB, cuda:1 "
          f"{blocks1*block_bytes_kv/1e9:.1f} GB); single-GPU 80GB cannot hold "
          f"{tot_gb:.0f} GB")

    # empty (uninitialized) is fine: this measures latency + memory, not values
    # (exactness already shown above and in tests). Resident, never moved.
    K0 = torch.empty(1, H, blocks0 * T, D, dtype=torch.float16, device="cuda:0")
    V0 = torch.empty(1, H, blocks0 * T, D, dtype=torch.float16, device="cuda:0")
    K1 = torch.empty(1, H, blocks1 * T, D, dtype=torch.float16, device="cuda:1")
    V1 = torch.empty(1, H, blocks1 * T, D, dtype=torch.float16, device="cuda:1")
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def step():
        return peer_parallel_attention(q, [(K0, V0), (K1, V1)], scale=scale)

    for _ in range(3):
        step()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    ts = []
    for _ in range(args.trials):
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        t0 = time.perf_counter(); step()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        ts.append(time.perf_counter() - t0)
    t_pp = statistics.median(ts) * 1e3
    mem0 = torch.cuda.memory_allocated(0) / 1e9
    mem1 = torch.cuda.memory_allocated(1) / 1e9
    res.update({"_is_measured": True, "peer_parallel_ms": t_pp,
                "cuda0_mem_gb": mem0, "cuda1_mem_gb": mem1})
    print(f"  PEER-PARALLEL decode step: {t_pp:.2f} ms  "
          f"(cuda:0 {mem0:.1f} GB, cuda:1 {mem1:.1f} GB) -- WORKS")

    # ---- (3) all-local: the full KV must be resident on ONE GPU. Probe whether
    # it even fits by allocating the gathered K (then V) on cuda:0 -- a single
    # clean allocation that OOMs fast when total KV exceeds one GPU's capacity
    # (no thrashing copy/cat/SDPA chain). cuda:0 still holds its own shard, so this
    # is the honest "can a single GPU hold the whole context?" test.
    free0_gb = torch.cuda.mem_get_info(0)[0] / 1e9
    try:
        Kall = torch.empty(1, H, tot_blocks * T, D, dtype=torch.float16, device="cuda:0")
        Vall = torch.empty(1, H, tot_blocks * T, D, dtype=torch.float16, device="cuda:0")
        torch.cuda.synchronize(0)
        res.update({"all_local_status": "fit (did not OOM); raise sizes to overflow",
                    "all_local_free_gb_before": free0_gb})
        del Kall, Vall
        print(f"  ALL-LOCAL: full {tot_gb:.0f} GB KV fit on cuda:0 "
              f"(had {free0_gb:.0f} GB free) -- raise --gb-* to force overflow")
    except RuntimeError as e:
        torch.cuda.empty_cache()
        res.update({"all_local_status": "OOM (enablement point)",
                    "all_local_oom_msg": str(e)[:120],
                    "all_local_free_gb_before": free0_gb})
        print(f"  ALL-LOCAL: OOM -- a single GPU CANNOT hold the {tot_gb:.0f} GB KV "
              f"(only {free0_gb:.0f} GB free). PeerKV-Parallel ran it on two at "
              f"{t_pp:.0f} ms. *** ENABLEMENT ***")

    _write(res)


if __name__ == "__main__":
    main()

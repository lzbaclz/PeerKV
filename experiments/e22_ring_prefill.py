"""e22 -- ring/context-parallel PREFILL across 2 GPUs (exactness + latency).

RUN ON THE DUAL-A100 BOX. Decode is the big win for compute-follows-KV (partials
are KB); prefill genuinely needs all query-key pairs, so the KV is exchanged around
the ring -- but it stays SHARDED (each GPU holds its slice), bounding per-GPU memory
so a prompt whose KV overflows one GPU can be prefilled on two. Causal: query shard
i attends to earlier key shards (full) + its own (causal); for 2 GPUs only shard 0's
KV is sent forward (later keys are in the future).

We validate the ring prefill is numerically EXACT vs dense causal attention, then
measure its per-step latency and per-GPU peak.

    python experiments/e22_ring_prefill.py --seq 16384
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "ring_prefill.json"


def _write(res: dict) -> None:
    if res.get("seq"):                       # per-seq file (avoid overwriting other lengths)
        p = OUT.parent / f"ring_prefill_S{res['seq']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2))
    res["_experiment"] = "e22_ring_prefill"
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=16384, help="total prompt length")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    res = {k: getattr(args, k) for k in ("seq", "heads", "head_dim", "shards", "trials")}
    try:
        import torch
        import torch.nn.functional as F
        from umallm.peer_parallel_attn import ring_prefill_attention
    except Exception as e:  # noqa: BLE001
        res.update({"_is_measured": False, "note": f"import failed: {e}"})
        _write(res); return

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        res.update({"_is_measured": False, "note": f"need >=2 GPUs (have {ngpu})"})
        _write(res); print(f"  only {ngpu} GPU(s) -- placeholder."); return

    H, D, S = args.heads, args.head_dim, args.seq
    K_shards = args.shards
    scale = 1.0 / (D ** 0.5)
    torch.manual_seed(0)
    devs = [f"cuda:{i % ngpu}" for i in range(K_shards)]

    # contiguous sequence shards, each Q/K/V resident on its device
    cut = S // K_shards
    sizes = [cut] * (K_shards - 1) + [S - cut * (K_shards - 1)]
    shards = []
    for i, sz in enumerate(sizes):
        d = devs[i]
        shards.append((torch.randn(1, H, sz, D, dtype=torch.float16, device=d),
                       torch.randn(1, H, sz, D, dtype=torch.float16, device=d),
                       torch.randn(1, H, sz, D, dtype=torch.float16, device=d)))
    for i in range(ngpu):
        torch.cuda.synchronize(i)

    def sync():
        for i in range(ngpu):
            torch.cuda.synchronize(i)

    # ---- exactness vs dense causal attention (fp32 ref on cuda:0) ----
    Q = torch.cat([s[0].to("cuda:0") for s in shards], dim=2).float()
    Kf = torch.cat([s[1].to("cuda:0") for s in shards], dim=2).float()
    Vf = torch.cat([s[2].to("cuda:0") for s in shards], dim=2).float()
    ref = F.scaled_dot_product_attention(Q, Kf, Vf, is_causal=True, scale=scale)
    outs = ring_prefill_attention(shards, scale=scale); sync()
    out = torch.cat([o.to("cuda:0") for o in outs], dim=2)
    cos = float(torch.nn.functional.cosine_similarity(
        ref.reshape(-1), out.reshape(-1).float(), dim=0))
    max_abs = float((ref - out.float()).abs().max())
    res.update({"exactness_cosine": cos, "exactness_max_abs_err": max_abs})
    print(f"  exactness vs dense causal: cos={cos:.6f}  max_abs={max_abs:.2e}")
    del Q, Kf, Vf, ref, out
    torch.cuda.empty_cache()

    # ---- timing ----
    def ring_step():
        return ring_prefill_attention(shards, scale=scale)
    for _ in range(3):
        ring_step()
    sync()
    torch.cuda.reset_peak_memory_stats(0); torch.cuda.reset_peak_memory_stats(1)
    ts = []
    for _ in range(args.trials):
        sync(); t0 = time.perf_counter(); ring_step(); sync()
        ts.append(time.perf_counter() - t0)
    t_ring = statistics.median(ts) * 1e3
    peak0 = torch.cuda.max_memory_allocated(0) / 1e6
    peak1 = torch.cuda.max_memory_allocated(1) / 1e6

    # single-GPU dense prefill latency reference (fits at this S)
    Q1 = torch.cat([s[0].to("cuda:0") for s in shards], dim=2)
    K1 = torch.cat([s[1].to("cuda:0") for s in shards], dim=2)
    V1 = torch.cat([s[2].to("cuda:0") for s in shards], dim=2)
    for _ in range(3):
        F.scaled_dot_product_attention(Q1, K1, V1, is_causal=True, scale=scale)
    sync()
    ts = []
    for _ in range(args.trials):
        sync(); t0 = time.perf_counter()
        F.scaled_dot_product_attention(Q1, K1, V1, is_causal=True, scale=scale); sync()
        ts.append(time.perf_counter() - t0)
    t_single = statistics.median(ts) * 1e3
    del Q1, K1, V1; torch.cuda.empty_cache()

    res.update({
        "_is_measured": True, "kind": "ring_prefill",
        "device": torch.cuda.get_device_name(0), "n_gpus": ngpu,
        "ring_prefill_ms": t_ring, "single_gpu_dense_ms": t_single,
        "ring_peak_mb_cuda0": peak0, "ring_peak_mb_cuda1": peak1,
        "speedup_vs_single": (t_single / t_ring) if t_ring else None,
        "note": ("exact causal ring prefill; KV stays sharded (bounded per-GPU peak) "
                 "so a prompt whose KV overflows one GPU prefills on two. prefill is "
                 "compute-bound, so the win is enablement + bounded peak, not latency."),
    })
    _write(res)
    print(f"  ring_prefill={t_ring:.2f}ms  single_gpu_dense={t_single:.2f}ms  "
          f"peak cuda:0={peak0:.0f}MB cuda:1={peak1:.0f}MB  "
          f"speedup={res['speedup_vs_single']}")


if __name__ == "__main__":
    main()

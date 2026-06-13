"""e36 -- Tree/DistAttention surrogate: the avoided-distribution regret on FITTING.

Head-to-head note (verified): PeerKV's merge_partial IS Tree Attention's logsumexp
reduction == DistAttention's MicroAttention reduce (cos=1.0, same FlashAttention-2
per shard). So a *mechanism* race is a tie by construction -- don't run it. The
real, measurable difference is the GATING POLICY: Tree always-reduces and
DistAttention distributes reactively under memory pressure; NEITHER has a deadline
gate to DECLINE distribution. PeerKV declines it for any request that fits one GPU.

This experiment quantifies exactly what PeerKV's gate avoids: at contexts that FIT
one GPU, we force the per-request partial-exchange (the Tree/DistAttention behavior)
and compare to running single-GPU (PeerKV's choice). The ratio is the avoided
regret -- the round-trip + setup tax paid on a request that never needed two GPUs.
Both arms are numerically exact (cos=1.0). This is the N=2 instantiation of their
mechanism + a gate; it is NOT a claim to beat their published cluster systems.

    /home/lzq/miniconda3/envs/peerkv/bin/python experiments/e36_tree_surrogate.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import peer_parallel_attention

    if torch.cuda.device_count() < 2:
        print("need >=2 GPUs"); return

    D = 128
    GEOMS = {"MHA": (32, 32), "GQA": (32, 8)}     # (q_heads, kv_heads)
    FIT_CTX = [2048, 4096, 8192, 16384, 32768]
    trials, seeds, warmup = 20, 3, 5
    scale = 1.0 / (D ** 0.5)

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med_iqr(xs):
        xs = sorted(xs)
        return statistics.median(xs), xs[(3*len(xs))//4] - xs[len(xs)//4]

    def timed(step):
        ms = []
        for _ in range(seeds):
            for _ in range(warmup):
                step()
            sync()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            for _ in range(trials):
                sync(); s.record(); step(); e.record(); sync()
                ms.append(s.elapsed_time(e))
        return med_iqr(ms)

    out = {
        "_experiment": "e36_tree_surrogate",
        "_is_measured": True,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "head_dim": D, "fit_contexts": FIT_CTX,
        "claim": ("Avoided-distribution regret: at contexts that FIT one GPU, the "
                  "Tree/DistAttention always-distribute behavior pays a round-trip the "
                  "PeerKV gate declines (runs single). Ratio = tree_surrogate/single. "
                  "Both exact (cos=1.0). N=2 instantiation + gate, not beating their "
                  "published systems."),
        "geometries": {},
    }

    for gname, (Hq, Hkv) in GEOMS.items():
        rows = []
        for ctx in FIT_CTX:
            half = ctx // 2
            q = torch.randn(1, Hq, 1, D, dtype=torch.float16, device="cuda:0")
            # single-GPU: full KV on cuda:0
            K = torch.randn(1, Hkv, ctx, D, dtype=torch.float16, device="cuda:0")
            V = torch.randn(1, Hkv, ctx, D, dtype=torch.float16, device="cuda:0")
            # tree-surrogate: KV split 50/50 across the two GPUs
            K0, V0 = K[:, :, :half, :].contiguous(), V[:, :, :half, :].contiguous()
            K1 = K[:, :, half:, :].contiguous().to("cuda:1")
            V1 = V[:, :, half:, :].contiguous().to("cuda:1")
            sync()

            def single_step():
                Kx = K.repeat_interleave(Hq // Hkv, dim=1) if Hkv != Hq else K
                Vx = V.repeat_interleave(Hq // Hkv, dim=1) if Hkv != Hq else V
                return F.scaled_dot_product_attention(q, Kx, Vx, scale=scale)

            def tree_step():
                return peer_parallel_attention(q, [(K0, V0), (K1, V1)], scale)

            # exactness
            with torch.no_grad():
                ref = single_step().float()
                tre = tree_step().float()
                cos = float(F.cosine_similarity(ref.reshape(-1), tre.reshape(-1), dim=0))

            sm, si = timed(single_step)
            tm, ti = timed(tree_step)
            rows.append({
                "ctx": ctx, "single_ms": round(sm, 4), "tree_surrogate_ms": round(tm, 4),
                "avoided_regret_x": round(tm / sm, 3), "cos": round(cos, 6),
            })
            print(f"[{gname}] ctx={ctx:6d}  single={sm:.4f}ms  tree={tm:.4f}ms  "
                  f"avoided_regret={tm/sm:.2f}x  cos={cos:.6f}")
            del K, V, K0, V0, K1, V1, q; torch.cuda.empty_cache()
        out["geometries"][gname] = {
            "q_heads": Hq, "kv_heads": Hkv, "rows": rows,
            "mean_avoided_regret_x": round(statistics.mean(r["avoided_regret_x"] for r in rows), 3),
        }

    path = os.path.join(RES, "tree_surrogate.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nmean avoided-distribution regret (Tree/DistAttn on fitting requests):")
    for g, d in out["geometries"].items():
        print(f"  {g}: {d['mean_avoided_regret_x']}x")
    print("wrote", path)


if __name__ == "__main__":
    main()

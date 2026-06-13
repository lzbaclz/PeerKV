"""e45 -- port-arbitration law vs Llama-3-8B-geometry decode holder (random weights).

Overlap-safe timing for push/pull under concurrent decode load.
"""
from __future__ import annotations
import argparse, json, statistics
from datetime import datetime, timezone
from pathlib import Path

from overlap_safe_bw import drain_stream, measure_bw_events, start_sustained_load

OUT = Path(__file__).resolve().parent / "results" / "port_arbitration_realdecode.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    dev = "cuda:1"
    torch.cuda.set_device(1)
    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = args.batch, args.ctx
    dt = torch.float16
    Wq = torch.randn(D, H * HD, dtype=dt, device=dev) * 0.02
    Wk = torch.randn(D, HKV * HD, dtype=dt, device=dev) * 0.02
    Wv = torch.randn(D, HKV * HD, dtype=dt, device=dev) * 0.02
    Wo = torch.randn(H * HD, D, dtype=dt, device=dev) * 0.02
    Wg = torch.randn(D, DFF, dtype=dt, device=dev) * 0.02
    Wu = torch.randn(D, DFF, dtype=dt, device=dev) * 0.02
    Wd = torch.randn(DFF, D, dtype=dt, device=dev) * 0.02
    Kc = torch.randn(B, HKV, S, HD, dtype=dt, device=dev) * 0.02
    Vc = torch.randn(B, HKV, S, HD, dtype=dt, device=dev) * 0.02
    x = torch.randn(B, 1, D, dtype=dt, device=dev) * 0.02

    def attn_step():
        q = (x @ Wq).view(B, 1, H, HD).transpose(1, 2)
        k = (x @ Wk).view(B, 1, HKV, HD).transpose(1, 2)
        v = (x @ Wv).view(B, 1, HKV, HD).transpose(1, 2)
        K = torch.cat([Kc, k], dim=2)
        V = torch.cat([Vc, v], dim=2)
        o = F.scaled_dot_product_attention(q, K, V, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ Wo)

    def ffn_step():
        return (F.silu(x @ Wg) * (x @ Wu)) @ Wd

    def full_step():
        return ffn_step() + attn_step()

    def per_iter_us(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize(1)
        import time
        t0 = time.perf_counter()
        for _ in range(50):
            fn()
        torch.cuda.synchronize(1)
        return (time.perf_counter() - t0) / 50 * 1e6

    torch.cuda.set_device(0)
    n = 256 * 1024 * 1024 // 2
    nbytes = n * 2
    src1 = torch.ones(n, dtype=dt, device="cuda:1")
    dst0 = torch.empty(n, dtype=dt, device="cuda:0")
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)
    holder_ls = torch.cuda.Stream(device=1)

    def pull():
        with torch.cuda.stream(s0):
            dst0.copy_(src1, non_blocking=True)

    def push():
        with torch.cuda.stream(s1):
            dst0.copy_(src1, non_blocking=True)

    pull_idle = statistics.mean(
        measure_bw_events(pull, s0, nbytes, args.trials)[0] for _ in range(args.seeds)
    )
    push_idle = statistics.mean(
        measure_bw_events(push, s1, nbytes, args.trials)[0] for _ in range(args.seeds)
    )

    out = {
        "_experiment": "e45_real_decode_holder",
        "_is_measured": True,
        "_timing_method": "overlap_safe_cuda_events",
        "device": torch.cuda.get_device_name(0),
        "geometry": "Llama-3-8B GQA (random weights)",
        "ctx": S,
        "batch": B,
        "seeds": args.seeds,
        "pull_idle_gbs": round(pull_idle, 1),
        "push_idle_gbs": round(push_idle, 1),
        "phases": {},
    }

    for name, fn in [("attention", attn_step), ("ffn", ffn_step), ("full_decode", full_step)]:
        per = per_iter_us(fn)
        pull_rets, push_rets, advs = [], [], []
        pull_gbs, push_gbs = [], []
        for _ in range(args.seeds):
            start_sustained_load(fn, holder_ls)
            pm, _ = measure_bw_events(
                pull, s0, nbytes, args.trials,
                holder_stream=holder_ls, holder_op=fn, assert_holder_busy=True,
            )
            pum, _ = measure_bw_events(
                push, s1, nbytes, args.trials,
                holder_stream=holder_ls, holder_op=fn, assert_holder_busy=True,
            )
            drain_stream(holder_ls)
            pull_rets.append(pm / pull_idle)
            push_rets.append(pum / push_idle)
            advs.append((pum / pm - 1) * 100)
            pull_gbs.append(pm)
            push_gbs.append(pum)

        phase = {
            "holder_per_iter_us": round(per, 1),
            "pull_busy_gbs": round(statistics.mean(pull_gbs), 1),
            "push_busy_gbs": round(statistics.mean(push_gbs), 1),
            "pull_retained": round(statistics.mean(pull_rets), 3),
            "push_retained": round(statistics.mean(push_rets), 3),
            "push_adv_pp": round(statistics.mean(advs), 1),
            "pull_retained_std": round(statistics.pstdev(pull_rets), 3) if args.seeds > 1 else 0,
            "push_retained_std": round(statistics.pstdev(push_rets), 3) if args.seeds > 1 else 0,
        }
        out["phases"][name] = phase
        print(f"  {name:13s} per_iter={per:6.0f}us  pull_ret={phase['pull_retained']:.2f}  "
              f"push_ret={phase['push_retained']:.2f}  push_adv=+{phase['push_adv_pp']:.0f}pp")

    fd = out["phases"]["full_decode"]
    out["verdict"] = (
        "LAW HOLDS on decode geometry"
        if fd["pull_retained"] < 0.9 and fd["push_adv_pp"] > 5
        else "LAW WEAK on decode geometry -> fall back to characterization"
    )
    out["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(out, indent=2))
    print(f"  idle: pull {pull_idle:.0f} push {push_idle:.0f} GB/s")
    print(f"  VERDICT: {out['verdict']}")
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

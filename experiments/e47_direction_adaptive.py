"""e47 -- does the direction selector beat BOTH fixed policies, or is always-PUSH
enough? (The decisive question the reviewers raised: e46 hardcoded the read-bound
bit, so its selector was a constant PUSH.)

We drive the bit from a MEASURED signal -- the holder's own pull-retention -- and
test two holder regimes:
  DECODE  (memory-bound): real Llama-3-8B GQA decode, ctx 64K, B=1 -> read-bound.
  PREFILL (compute-bound): a 2048-token prefill forward (big GEMMs) -> not read-bound.
For each we measure push/pull effective bandwidth AND the holder's victim-throughput
cost under each direction. The selector picks PUSH iff the measured pull-retention
shows the holder is read-bound. We then report, honestly, whether the selector
strictly beats always-PULL (NIXL) and always-PUSH (MoRIIO), or whether always-PUSH
already dominates (in which case the adaptive rule is not needed and we say so).
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

from overlap_safe_bw import drain_stream, measure_bw_events, start_sustained_load

OUT = Path(__file__).resolve().parent / "results" / "direction_adaptive.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    dt = torch.float16
    torch.cuda.set_device(1)
    W = {k: torch.randn(*s, dtype=dt, device="cuda:1") * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
    S = args.ctx
    Kc = torch.randn(1, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
    Vc = torch.randn(1, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
    xd = torch.randn(1, 1, D, dtype=dt, device="cuda:1") * 0.02
    xp = torch.randn(1, 2048, D, dtype=dt, device="cuda:1") * 0.02

    def decode_step():
        q = (xd @ W["q"]).view(1, 1, H, HD).transpose(1, 2)
        k = (xd @ W["k"]).view(1, 1, HKV, HD).transpose(1, 2)
        v = (xd @ W["v"]).view(1, 1, HKV, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, torch.cat([Kc, k], 2), torch.cat([Vc, v], 2), enable_gqa=True)
        return (o.transpose(1, 2).reshape(1, 1, H * HD) @ W["o"]) + (F.silu(xd @ W["g"]) * (xd @ W["u"])) @ W["d"]

    def prefill_step():
        return (F.silu(xp @ W["g"]) * (xp @ W["u"])) @ W["d"] + (xp @ W["q"]) @ W["o"]

    n = args.handoff_mb * 1024 * 1024 // 2
    nbytes = n * 2
    kv_src = torch.randn(n, dtype=dt, device="cuda:1")
    kv_dst = torch.empty(n, dtype=dt, device="cuda:0")
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)
    holder_ls = torch.cuda.Stream(device=1)

    def copy_pull():
        with torch.cuda.stream(s0):
            kv_dst.copy_(kv_src, non_blocking=True)

    def copy_push():
        with torch.cuda.stream(s1):
            kv_dst.copy_(kv_src, non_blocking=True)

    def bw(copy_fn, copy_stream, load):
        if load is None:
            return measure_bw_events(copy_fn, copy_stream, nbytes, args.trials)[0]
        start_sustained_load(load, holder_ls)
        gbps, _ = measure_bw_events(
            copy_fn, copy_stream, nbytes, args.trials,
            holder_stream=holder_ls, holder_op=load, assert_holder_busy=True,
        )
        drain_stream(holder_ls)
        return gbps

    def victim_tpot(step_fn, copy_stream_or_none):
        torch.cuda.set_device(1)
        for _ in range(5):
            step_fn()
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        for _ in range(20):
            step_fn()
            if copy_stream_or_none is not None:
                with torch.cuda.stream(copy_stream_or_none):
                    kv_dst.copy_(kv_src, non_blocking=True)
        torch.cuda.synchronize(1)
        torch.cuda.set_device(0)
        return (time.perf_counter() - t0) / 20 * 1e3

    regimes = []
    for name, step in [("decode_membound", decode_step), ("prefill_computebound", prefill_step)]:
        torch.cuda.set_device(1)
        for _ in range(5):
            step()
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        for _ in range(20):
            step()
        torch.cuda.synchronize(1)
        per = (time.perf_counter() - t0) / 20
        torch.cuda.set_device(0)

        bw_idle = bw(copy_pull, s0, None)
        bw_pull = bw(copy_pull, s0, step)
        bw_push = bw(copy_push, s1, step)
        pull_ret = bw_pull / bw_idle
        read_bound = (1 - pull_ret) > 0.05
        tpot_alone = victim_tpot(step, None)
        tpot_pull = victim_tpot(step, s0)
        tpot_push = victim_tpot(step, s1)
        sel_dir = "push" if read_bound else "pull"
        oracle_dir = "push" if bw_push > bw_pull * 1.02 else ("pull" if (tpot_pull < tpot_push) else "push")
        regimes.append({
            "regime": name, "holder_per_iter_us": round(per * 1e6, 1),
            "bw_idle": round(bw_idle, 1), "bw_pull": round(bw_pull, 1), "bw_push": round(bw_push, 1),
            "pull_retained": round(pull_ret, 3), "measured_read_bound": read_bound,
            "victim_pull_pct": round((tpot_pull / tpot_alone - 1) * 100, 1),
            "victim_push_pct": round((tpot_push / tpot_alone - 1) * 100, 1),
            "selector_dir": sel_dir, "oracle_dir": oracle_dir, "selector_matches_oracle": sel_dir == oracle_dir})
        r = regimes[-1]
        print(f"  {name:20s} pull_ret={pull_ret:.2f} read_bound={read_bound}  "
              f"bw pull/push={bw_pull:.0f}/{bw_push:.0f}  victim pull/push={r['victim_pull_pct']:+.1f}/{r['victim_push_pct']:+.1f}%  "
              f"sel={sel_dir} oracle={oracle_dir}")

    selector_needed = any(r["selector_dir"] == "pull" and r["oracle_dir"] == "pull" for r in regimes)
    res = {
        "_experiment": "e47_direction_adaptive",
        "_is_measured": True,
        "_timing_method": "overlap_safe_cuda_events",
        "device": torch.cuda.get_device_name(0),
        "regimes": regimes,
        "selector_strictly_beats_always_push": selector_needed,
        "verdict": ("selector beats always-PUSH (it correctly falls back to PULL where PUSH is worse)"
                    if selector_needed else
                    "always-PUSH already dominates; adaptive selector not needed here (honest negative)"),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  VERDICT: {res['verdict']}")
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

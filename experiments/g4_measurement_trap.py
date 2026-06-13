"""G4 -- the measurement trap: how naive timing manufactures a phantom push>pull
asymmetry that vanishes under controlled measurement.

We measure the SAME workload (holder GPU1 decoding + a concurrent 512MB cross-GPU
handoff, push vs pull) with two timing methods:

  naive : wall-clock around `for i: decode(default stream); one 512MB copy(copy stream)`
          ended by torch.cuda.synchronize(1)  (== the legacy e46/e47 victim_tpot)
  clean : per-iteration CUDA events on the decode stream, copy kept saturated on its
          own stream, never a whole-device sync   (== G1/G2)

Run this once with clocks UNLOCKED and once LOCKED (see g4_run.sh). The phantom
push/pull asymmetry appears under `naive` (worst with unlocked clocks) and collapses
under `clean+locked`. This quantifies the artifact behind the retracted +30pp / +146%
claims and is the paper's methodology contribution.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", type=str, default="locked", help="clock state label for the output")
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g4_measurement_trap")
    import torch
    import torch.nn.functional as F
    import pynvml
    pynvml.nvmlInit()
    clk = []
    for i in (0, 1):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        clk.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS))

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = 1, args.ctx
    dt = torch.float16
    torch.cuda.set_device(1)
    W = {k: torch.randn(*s, dtype=dt, device="cuda:1") * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
    Kc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
    Vc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
    x = torch.randn(B, 1, D, dtype=dt, device="cuda:1") * 0.02

    def decode_step():
        q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
        k = (x @ W["k"]).view(B, 1, HKV, HD).transpose(1, 2)
        v = (x @ W["v"]).view(B, 1, HKV, HD).transpose(1, 2)
        # faithful memory-bound decode: flash-attend the resident L-token KV in place
        # (paged-decode HBM-read pattern), no per-step full re-cat of the cache.
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    total = args.handoff_mb * 1024 * 1024 // 2
    src_h = torch.randn(total, dtype=dt, device="cuda:1")
    dst_c = torch.empty(total, dtype=dt, device="cuda:0")
    s_dev1 = torch.cuda.Stream(device=1)
    s_dev0 = torch.cuda.Stream(device=0)
    dec_stream = torch.cuda.Stream(device=1)

    def copy_stream(direction):
        return s_dev1 if direction == "push" else s_dev0

    # warm
    torch.cuda.set_device(1)
    for _ in range(20):
        decode_step()
    torch.cuda.synchronize(1)

    def naive_victim(direction):
        """Legacy e47-style: default-stream decode + one copy/step, whole-device sync."""
        cs = copy_stream(direction)
        torch.cuda.set_device(1)
        for _ in range(5):
            decode_step()
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            decode_step()
            with torch.cuda.stream(cs):
                dst_c.copy_(src_h, non_blocking=True)
        torch.cuda.synchronize(1)
        t_with = (time.perf_counter() - t0) / args.iters
        # baseline alone
        t0 = time.perf_counter()
        for _ in range(args.iters):
            decode_step()
        torch.cuda.synchronize(1)
        t_alone = (time.perf_counter() - t0) / args.iters
        return (t_with / t_alone - 1) * 100

    def clean_victim(direction):
        """G1/G2-style: per-iter events on decode stream, saturated copy, no device sync."""
        cs = copy_stream(direction)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)

        def timed():
            with torch.cuda.stream(dec_stream):
                e0.record(dec_stream); decode_step(); e1.record(dec_stream)
            e1.synchronize()
            return e0.elapsed_time(e1)
        base = statistics.median([timed() for _ in range(args.iters)])

        def op():
            with torch.cuda.stream(cs):
                dst_c.copy_(src_h, non_blocking=True)
        for _ in range(3):
            op()
        during = []
        for _ in range(args.iters):
            if cs.query():
                op()
            during.append(timed())
        return (statistics.median(during) / base - 1) * 100

    res = {"push": {}, "pull": {}}
    for method, fn in [("naive", naive_victim), ("clean", clean_victim)]:
        for direction in ("push", "pull"):
            vals = [fn(direction) for _ in range(args.seeds)]
            res[direction][method] = {"victim_pct": round(statistics.mean(vals), 2),
                                      "std": round(statistics.pstdev(vals), 2) if args.seeds > 1 else 0.0}
    for method in ("naive", "clean"):
        p = res["push"][method]["victim_pct"]; q = res["pull"][method]["victim_pct"]
        print(f"  [{args.tag:8s}] {method:5s}: push +{p:.1f}%  pull +{q:.1f}%  "
              f"phantom_asymmetry(pull-push)={q - p:+.1f}pp")

    out = {"_experiment": "g4_measurement_trap", "_is_measured": True,
           "clock_state": args.tag, "graphics_clock_mhz": clk,
           "device": torch.cuda.get_device_name(0), "ctx": S, "handoff_mb": args.handoff_mb,
           "results": res, "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"g4_trap_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RESULTS / f'g4_trap_{args.tag}.json'}")


if __name__ == "__main__":
    main()

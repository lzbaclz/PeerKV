"""S3 -- feedforward quote accuracy on held-out random operating points.

The governor's fast path is open-loop: it admits against the s1 curves
without waiting for feedback.  S3 asks the only question that matters for
that path: at operating points the census never visited, how far is
quote()'s predicted victim cost from the measured one?

Procedure: draw random (rate, victim batch) pairs on the receiver/peer route
-- the polarity that broke the injector curve (g11's 2x) -- measure the real
victim slowdown under a sustained paced train at that rate (same protocol as
s1), and compare against the calibration's central and upper predictions for
the matching workload bucket AND for the 'worst' bucket the hint-less
governor actually uses.  The reported residual distribution IS the floor of
the enforceable eps for open-loop admission (critique: "set the enforceable
eps floor at the measured residual of the law").
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _s_common import Train, make_decoder  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "s3_quote_accuracy.json"
CALIB = RESULTS / "governor_calib.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--probes", type=int, default=14)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--buf-mb", type=int, default=512)
    ap.add_argument("--train-secs", type=float, default=4.0)
    ap.add_argument("--warm-secs", type=float, default=0.6)
    ap.add_argument("--base-iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260610)
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s3_quote_accuracy")
    import torch
    from umallm.governor.pacer import PacedCopier
    from umallm.governor.calib import GovernorCalibration

    assert torch.cuda.device_count() >= 2
    assert CALIB.exists(), "run s1_governor_calib.py first"
    cal = GovernorCalibration.load(CALIB)
    link = cal.link_peak_gbs["peer"]

    dt = torch.float16
    n = args.buf_mb * (1 << 20) // 2
    src = torch.randn(n, dtype=dt, device="cuda:1")
    dst = torch.empty(n, dtype=dt, device="cuda:0")
    copier = PacedCopier(device=1)
    train = Train(copier, src, dst, args.chunk_mb << 20)

    rng = random.Random(args.seed)
    probes = []
    for batch in args.batches:
        timed_iter = make_decoder("cuda:0", batch, args.ctx)
        base = statistics.median([timed_iter() for _ in range(args.base_iters)])
        for i in range(args.probes // len(args.batches)):
            # stratified: half the probes inside the admission region
            # (<= ~0.27 x link ~= the eps=5% inverted caps), half above --
            # an unstratified draw left the admitted band with ZERO coverage
            rate = (rng.uniform(0.08, 0.27) if i % 2 == 0
                    else rng.uniform(0.27, 1.0)) * link
            train.start(rate)
            time.sleep(args.warm_secs)
            t_end = time.monotonic() + args.train_secs
            during = []
            while time.monotonic() < t_end:
                during.append(timed_iter())
            achieved = train.stop()
            med = statistics.median(during)
            measured = (med / base - 1) * 100
            wl = f"b{batch}"
            c_match = cal.curve("receiver", "peer", wl)
            c_worst = cal.curve("receiver", "peer", "worst")
            probes.append({
                "workload": wl, "target_rate_gbs": round(rate, 1),
                "achieved_rate_gbs": round(achieved, 1),
                "measured_pct": round(measured, 2),
                "pred_central_pct": round(c_match.predict(achieved), 2),
                "pred_upper_pct": round(c_match.predict_upper(achieved), 2),
                "pred_worst_upper_pct": round(c_worst.predict_upper(achieved), 2),
                "iters": len(during),
            })
            p = probes[-1]
            print(f"  {wl} rate={p['achieved_rate_gbs']:6.1f}GB/s "
                  f"measured=+{p['measured_pct']:6.2f}% "
                  f"central=+{p['pred_central_pct']:6.2f}% "
                  f"upper=+{p['pred_upper_pct']:6.2f}% "
                  f"worst_upper=+{p['pred_worst_upper_pct']:6.2f}%")
    train.shutdown()

    def block(ps):
        resid = [p["measured_pct"] - p["pred_central_pct"] for p in ps]
        under = [p["measured_pct"] - p["pred_worst_upper_pct"] for p in ps]
        return {
            "n_probes": len(ps),
            "central_residual_pp": {
                "mean": round(statistics.mean(resid), 2),
                "max_abs": round(max(abs(r) for r in resid), 2),
                "p50_abs": round(statistics.median(abs(r) for r in resid), 2),
            },
            "worst_envelope_exceedance_pp": {
                "max": round(max(under), 2),
                "frac_above": round(
                    sum(1 for u in under if u > 0) / len(under), 3),
            },
        }

    ADMIT_GBS = 70.0   # the eps=5% inverted caps land near 50; <=70 is the band
    summary = {
        "all": block(probes),
        "admission_region": block(
            [p for p in probes if p["achieved_rate_gbs"] <= ADMIT_GBS]),
        "above_region": block(
            [p for p in probes if p["achieved_rate_gbs"] > ADMIT_GBS]),
        "admission_region_cutoff_gbs": ADMIT_GBS,
        "note": ("worst_envelope_exceedance.max in the admission region is "
                 "the open-loop eps floor (a LOWER-BOUND estimate at this n; "
                 "exceedances cluster in census knot gaps)"),
    }
    out = {
        "_experiment": "s3_quote_accuracy",
        "_is_measured": True,
        "_timing_method": "same protocol as s1; held-out random rates",
        "device": torch.cuda.get_device_name(0),
        "ctx": args.ctx, "probes_per_batch": args.probes // len(args.batches),
        "train_secs": args.train_secs, "seed": args.seed,
        "summary": summary, "probes": probes,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

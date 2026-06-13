"""S7 -- native (C++) pacer validation: closes the review's two open majors.

Three sub-experiments:

A. COMMAND TRACKING A/B (python vs native actuator, no victim): sustained
   trains at commanded rates across the full link range.  The Python pacer's
   measured undershoot (10-27% above 0.5x link) capped the s1 census domain
   at 0.72x link; the native pacer (GIL-free C++ loop, coarse-sleep+spin
   gaps) should track to ~1-2% everywhere.

B. MATCHED-CADENCE NULL (the GIL-confound control the measurement reviewer
   demanded): pace NEAR-ZERO-FOOTPRINT 64 KB chunks at the same launches/s
   cadence as each s1 census point (64 MB chunks at 31-190 GB/s = ~480-3000
   launches/s) while the b1 victim decodes.  If the census slowdowns were
   fabricated by host-launch/GIL activity rather than HBM footprint, this
   arm reproduces them; if the law is real, victim slowdown stays ~0.
   Run for BOTH pacers: the python row validates the already-published s1
   numbers, the native row shows the confound is gone by construction.

C. CENSUS DOMAIN EXTENSION (native only): receiver/peer at 0.85x and 1.0x
   link with the b1/b8 victims -- the region the python actuator could not
   reach, where the old curve clamped (predict_upper(261)=20.3%) while g11's
   burst anchor says ~23.5%.  Output feeds the curve-top extension.

Protocol: as s1 (victim-stream events, idle-machinery baseline, no
whole-device sync, locked clocks, box gated).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _s_common import Train, make_decoder  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "s7_native_pacer.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--buf-mb", type=int, default=512)
    ap.add_argument("--track-secs", type=float, default=3.0)
    ap.add_argument("--null-secs", type=float, default=4.0)
    ap.add_argument("--ext-secs", type=float, default=4.0)
    ap.add_argument("--ext-reps", type=int, default=3)
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s7_native_pacer")
    import torch
    from umallm.governor.pacer import PacedCopier
    from umallm.governor.native import NativePacedCopier

    assert torch.cuda.device_count() >= 2
    dt = torch.float16
    n = args.buf_mb * (1 << 20) // 2
    src = torch.randn(n, dtype=dt, device="cuda:1")
    dst = torch.empty(n, dtype=dt, device="cuda:0")
    chunk_bytes = args.chunk_mb << 20
    out: dict = {}

    # ---------- A. command tracking ---------------------------------------
    def link_rate(copier) -> float:
        for _ in range(2):
            copier.run(src, dst, lambda: 0.0, chunk_bytes, unpaced=True)
        return statistics.median(
            copier.run(src, dst, lambda: 0.0, chunk_bytes,
                       unpaced=True).achieved_gbs for _ in range(3))

    copiers = {"python": PacedCopier(device=1),
               "native": NativePacedCopier(device=1)}
    link = link_rate(copiers["native"])
    print(f"link (native unpaced): {link:.1f} GB/s")
    track = []
    for kind, copier in copiers.items():
        train = Train(copier, src, dst, chunk_bytes)
        for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
            cmd = frac * link
            train.start(cmd)
            time.sleep(args.track_secs)
            ach = train.stop()
            track.append({"pacer": kind, "commanded_gbs": round(cmd, 1),
                          "achieved_gbs": round(ach, 1),
                          "tracking_err_pct": round((ach / cmd - 1) * 100, 1)})
            print(f"  A {kind:6s} cmd={cmd:6.1f} ach={ach:6.1f} "
                  f"err={track[-1]['tracking_err_pct']:+5.1f}%")
        train.shutdown()
    out["A_tracking"] = track

    # ---------- B. matched-cadence null ------------------------------------
    null_chunk = 64 * 1024
    timed_iter = make_decoder("cuda:0", 1, args.ctx)
    null_rows = []
    for kind, mk in (("python", lambda: PacedCopier(device=1)),
                     ("native", lambda: NativePacedCopier(device=1))):
        copier = mk()
        train = Train(copier, src[: null_chunk * 8 // 2],
                      dst[: null_chunk * 8 // 2], null_chunk)
        base = statistics.median([timed_iter() for _ in range(200)])
        for cadence in (500, 1500, 3000):           # launches per second
            rate = cadence * null_chunk / 1e9        # GB/s for 64KB chunks
            train.start(rate)
            time.sleep(0.4)
            t_end = time.monotonic() + args.null_secs
            during = []
            while time.monotonic() < t_end:
                during.append(timed_iter())
            ach = train.stop()
            med = statistics.median(during)
            during.sort()
            row = {"pacer": kind, "cadence_per_s": cadence,
                   "footprint_gbs": round(ach, 3),
                   "victim_med_slowdown_pct": round((med / base - 1) * 100, 2),
                   "victim_p99_slowdown_pct": round(
                       (during[int(0.99 * len(during))] / base - 1) * 100, 2),
                   "iters": len(during)}
            null_rows.append(row)
            print(f"  B {kind:6s} cadence={cadence:4d}/s "
                  f"footprint={ach:6.3f} GB/s "
                  f"victim med {row['victim_med_slowdown_pct']:+5.2f}% "
                  f"(p99 {row['victim_p99_slowdown_pct']:+5.2f}%)")
        train.shutdown()
    out["B_cadence_null"] = null_rows

    # ---------- C. census extension (native, full-rate region) -------------
    ext_rows = []
    copier = NativePacedCopier(device=1)
    train = Train(copier, src, dst, chunk_bytes)
    for batch in (1, 8):
        timed_iter = make_decoder("cuda:0", batch, args.ctx)
        for rep in range(args.ext_reps):
            base = statistics.median([timed_iter() for _ in range(200)])
            for frac in (0.85, 1.0):
                train.start(frac * link)
                time.sleep(0.5)
                t_end = time.monotonic() + args.ext_secs
                during = []
                while time.monotonic() < t_end:
                    during.append(timed_iter())
                ach = train.stop()
                med = statistics.median(during)
                row = {"workload": f"b{batch}", "rep": rep,
                       "target_gbs": round(frac * link, 1),
                       "achieved_gbs": round(ach, 1),
                       "victim_med_slowdown_pct": round(
                           (med / base - 1) * 100, 2)}
                ext_rows.append(row)
                print(f"  C b{batch} rep{rep} rate={ach:6.1f} GB/s "
                      f"victim +{row['victim_med_slowdown_pct']:.2f}%")
    train.shutdown()
    out["C_census_extension"] = ext_rows

    blob = {
        "_experiment": "s7_native_pacer",
        "_is_measured": True,
        "_timing_method": ("victim-decode-stream CUDA events; sustained "
                           "trains; idle-machinery baselines; no whole-device "
                           "sync; clocks locked"),
        "device": torch.cuda.get_device_name(0),
        "link_gbs_native_unpaced": round(link, 1),
        "ctx": args.ctx, "chunk_mb": args.chunk_mb,
        **out,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(blob, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

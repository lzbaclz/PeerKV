"""e2 -- Sizing frontier (Claim C2: the inversion is actionable).

Fully synthetic, no hardware needed: this driver produces *real numbers*
in the sandbox. It sweeps a grid of operator deadlines ``D`` against
deadline-miss targets ``rho`` and, for both the sub-Gaussian and Bernstein
tail bounds, calls
:meth:`umallm.uma_model.UMACostModel.min_active_blocks_for_slo` to obtain
the minimum active-tier budget that keeps ``Pr(C_t > D) <= rho``.

The point an ICCD reviewer will check is *"is the schedulability bound
vacuous?"*. We answer it by reporting, per (D, rho) cell:

* whether the cell is **actionable** -- feasible *and* the achieved
  ``bound_at_solution <= rho`` (a non-trivial budget exists), and
* the **deadline floor** (compute+attention-only latency); deadlines at
  or below it are correctly reported infeasible.

We also confirm the two structural properties the theory promises:
looser deadlines admit smaller budgets, and Bernstein is never less
conservative than sub-Gaussian.
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

from . import add_repo_to_path, save_result

add_repo_to_path()

from umallm.uma_model import UMACostModel  # noqa: E402

# Deadlines (ms) and miss targets per the task / ICCD plan.
DEADLINES_MS = [25, 35, 50, 75, 100, 200]
MISS_TARGETS = [1e-2, 1e-3]
MODES = ["subgaussian", "bernstein"]


def run(
    n_blocks: int = 512,
    ema_attention_lat_us: float = 10_000.0,
    ema_compute_lat_us: float = 15_000.0,
    deadlines_ms: list[int] | None = None,
    miss_targets: list[float] | None = None,
) -> dict:
    """Sweep (deadline, miss-target, mode) and tabulate actionable cells.

    A representative single-request working set: ``n_blocks`` KV blocks, a
    25 ms compute+attention floor (10 ms attention + 15 ms compute), so the
    25 ms deadline sits exactly on the floor and is expected infeasible.
    """
    deadlines_ms = deadlines_ms or DEADLINES_MS
    miss_targets = miss_targets or MISS_TARGETS
    model = UMACostModel()

    cells = []
    deadline_floor_us = None
    for mode in MODES:
        for rho in miss_targets:
            for d_ms in deadlines_ms:
                res = model.min_active_blocks_for_slo(
                    deadline_us=d_ms * 1000.0,
                    ema_attention_lat=ema_attention_lat_us,
                    ema_compute_lat=ema_compute_lat_us,
                    n_blocks=n_blocks,
                    miss_target=rho,
                    mode=mode,
                )
                deadline_floor_us = res.deadline_floor_us
                actionable = bool(res.feasible and res.bound_at_solution <= rho)
                cells.append(
                    {
                        "mode": mode,
                        "deadline_ms": d_ms,
                        "miss_target": rho,
                        "feasible": bool(res.feasible),
                        "actionable": actionable,
                        "min_active_blocks": int(res.min_active_blocks),
                        "max_slow_fraction": float(res.max_slow_fraction),
                        "bound_at_solution": float(res.bound_at_solution),
                    }
                )

    n_actionable = sum(1 for c in cells if c["actionable"])
    deadline_floor_ms = deadline_floor_us / 1000.0 if deadline_floor_us else None

    # Structural checks (reported, not just asserted) ---------------------- #
    # Monotonicity: within a (mode, rho), looser deadline -> <= min_active.
    monotone = True
    for mode in MODES:
        for rho in miss_targets:
            seq = [
                c["min_active_blocks"]
                for c in sorted(
                    (c for c in cells if c["mode"] == mode and c["miss_target"] == rho),
                    key=lambda c: c["deadline_ms"],
                )
            ]
            if any(b > a for a, b in zip(seq, seq[1:])):
                monotone = False
    # Bernstein >= sub-Gaussian budget at matched (D, rho).
    bernstein_conservative = True
    sg = {(c["deadline_ms"], c["miss_target"]): c["min_active_blocks"]
          for c in cells if c["mode"] == "subgaussian"}
    for c in cells:
        if c["mode"] == "bernstein":
            if c["min_active_blocks"] < sg.get((c["deadline_ms"], c["miss_target"]), 0):
                bernstein_conservative = False

    payload = {
        "_is_measured": True,  # fully synthetic closed-form result; real now
        "n_blocks": n_blocks,
        "ema_attention_lat_us": ema_attention_lat_us,
        "ema_compute_lat_us": ema_compute_lat_us,
        "deadline_floor_us": deadline_floor_us,
        "deadline_floor_ms": deadline_floor_ms,
        "deadlines_ms": deadlines_ms,
        "miss_targets": miss_targets,
        "cells": cells,
        "n_cells": len(cells),
        "n_actionable": n_actionable,
        "monotone_in_deadline": monotone,
        "bernstein_at_least_as_conservative": bernstein_conservative,
    }
    return payload


def main() -> dict:
    payload = run()
    print("=== e2 sizing frontier (synthetic, real numbers) ===")
    print(f"  deadline floor = {payload['deadline_floor_ms']:.1f} ms "
          f"(D at/below this is infeasible)")
    print(f"  actionable cells: {payload['n_actionable']}/{payload['n_cells']}")
    print("  mode        rho      D(ms)  feas  min_active  max_slow  bound")
    for c in payload["cells"]:
        print(
            f"  {c['mode']:11s} {c['miss_target']:.0e}  {c['deadline_ms']:5d}  "
            f"{str(c['feasible']):5s} {c['min_active_blocks']:10d}  "
            f"{c['max_slow_fraction']:.3f}    {c['bound_at_solution']:.2e}"
        )
    print(f"  monotone_in_deadline={payload['monotone_in_deadline']} "
          f"bernstein_conservative={payload['bernstein_at_least_as_conservative']}")
    path = save_result("e2_sizing_frontier", payload)
    print(f"  -> {path}")
    return payload


if __name__ == "__main__":
    main()

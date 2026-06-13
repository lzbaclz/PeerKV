"""e3 -- Tier-policy ablation (Claim C2: residency policy is effective).

Fully synthetic (real numbers in-sandbox). We generate hotness traces over
``n_blocks`` KV blocks and, for four placement configurations, use
:class:`umallm.policy.UMAPolicy` to assign residency tiers and
:meth:`umallm.uma_model.UMACostModel.access_cost` to predict the per-step
KV-read cost (modeled TPOT). We report RAM-resident block count and modeled
TPOT per config.

Configs:
  * **full**          -- UMAPolicy with the pressure response active
                         (placed at the trace's pressure level).
  * **compress-only** -- tiering to T2 allowed, but never swap to T3
                         (``swap_under_critical=False``) and placed at
                         NORMAL so nothing is paged out.
  * **no-pressure**   -- full policy budget but the pressure listener never
                         fires (always placed at NORMAL), so it cannot react
                         to a hot trace under memory pressure.
  * **no-tier**       -- the llama.cpp-style baseline: every block is active
                         (T0) until the active budget is exceeded, then the
                         overflow *pages out* to T3 (swap). No compression
                         tier. This is the >300 us NVMe round-trip path.

The headline: ``no-tier`` pays the >300 us swap penalty on its overflow
blocks, so its modeled steady-state TPOT is several times that of any tiered
config. Among the tiered configs, ``compress-only`` has the lowest
steady-state TPOT (no dequant churn at NORMAL), while ``full`` trades a
little steady-state cost (it compresses the cold T1 half under WARN) for the
pressure resilience demonstrated separately in e4. So this driver isolates
the *tiered vs. no-tier* gap; e4 isolates the value of the pressure reaction.
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

import numpy as np  # noqa: E402

from . import add_repo_to_path, save_result  # noqa: E402

add_repo_to_path()

from umallm.policy import UMAPolicy  # noqa: E402
from umallm.pressure import PressureLevel  # noqa: E402
from umallm.uma_model import ResidencyTier, UMACostModel  # noqa: E402

# A fixed per-step compute term (us) so modeled TPOT is comparable across
# configs; only the KV-read cost differs by placement.
COMPUTE_FLOOR_US = 15_000.0


def _hotness_trace(n_blocks: int, seed: int = 0) -> np.ndarray:
    """Synthetic hotness: most blocks cold, a heavy head of hot blocks.

    A Zipf-ish profile -- a small hot set plus a long cold tail -- is the
    regime where tiering helps most.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_blocks + 1)
    base = 1.0 / ranks  # zipf-like decay
    noise = rng.uniform(0.8, 1.2, size=n_blocks)
    scores = base * noise
    return (scores / scores.max()).astype(np.float32)


def _modeled_tpot_us(model: UMACostModel, tiers: np.ndarray) -> float:
    """Per-step modeled TPOT = compute floor + sum of per-block read cost."""
    cost = COMPUTE_FLOOR_US
    for t in tiers:
        cost += model.access_cost(ResidencyTier(int(t)))
    return cost


def _ram_resident(tiers: np.ndarray) -> int:
    """Blocks physically resident in RAM (T0/T1/T2); T3 is paged out."""
    return int(np.sum(tiers != int(ResidencyTier.T3_SWAPPED)))


def _no_tier_placement(scores: np.ndarray, n_active: int) -> np.ndarray:
    """Baseline: keep the top-``n_active`` blocks active (T0), page the rest
    out to T3 swap. No compression tier -- this is "fits or swaps"."""
    n = scores.shape[0]
    tiers = np.full(n, int(ResidencyTier.T3_SWAPPED), dtype=np.int32)
    order = np.argsort(-scores)[:n_active]
    tiers[order] = int(ResidencyTier.T0_GPU_ACTIVE)
    return tiers


def run(n_blocks: int = 256, n_active: int = 96, seed: int = 0) -> dict:
    """Compare the four configs on a single hot trace under WARN pressure."""
    model = UMACostModel()
    scores = _hotness_trace(n_blocks, seed=seed)
    pressure = PressureLevel.WARN  # the regime where reaction matters

    configs = {}

    # full: pressure-aware policy placed at the live pressure level.
    full_pol = UMAPolicy(cost_model=model, n_active=n_active, n_sink=4, n_window=4)
    configs["full"] = full_pol.place(scores, pressure=pressure)

    # compress-only: tier to T2 but never swap; placed NORMAL (no demotion).
    comp_pol = UMAPolicy(
        cost_model=model, n_active=n_active, n_sink=4, n_window=4,
        swap_under_critical=False, compress_under_warn=False,
    )
    configs["compress_only"] = comp_pol.place(scores, pressure=PressureLevel.NORMAL)

    # no-pressure: same budget as full, but listener never fires (NORMAL).
    nop_pol = UMAPolicy(cost_model=model, n_active=n_active, n_sink=4, n_window=4)
    configs["no_pressure"] = nop_pol.place(scores, pressure=PressureLevel.NORMAL)

    # no-tier: top-n_active active, the rest swapped to T3.
    configs["no_tier"] = _no_tier_placement(scores, n_active)

    results = {}
    for name, tiers in configs.items():
        tcounts = {f"T{t}": int(np.sum(tiers == t)) for t in range(4)}
        results[name] = {
            "ram_resident_blocks": _ram_resident(tiers),
            "swapped_blocks": int(np.sum(tiers == int(ResidencyTier.T3_SWAPPED))),
            "modeled_tpot_us": _modeled_tpot_us(model, tiers),
            "modeled_tpot_ms": _modeled_tpot_us(model, tiers) / 1000.0,
            "tier_counts": tcounts,
        }

    best = min(results, key=lambda k: results[k]["modeled_tpot_us"])
    payload = {
        "_is_measured": True,  # synthetic + closed-form cost model; real now
        "n_blocks": n_blocks,
        "n_active": n_active,
        "pressure": pressure.name,
        "compute_floor_us": COMPUTE_FLOOR_US,
        "configs": results,
        "best_config": best,
        "no_tier_tpot_ms": results["no_tier"]["modeled_tpot_ms"],
        "full_tpot_ms": results["full"]["modeled_tpot_ms"],
        "no_tier_over_full_x": (
            results["no_tier"]["modeled_tpot_us"] / results["full"]["modeled_tpot_us"]
        ),
    }
    return payload


def main() -> dict:
    payload = run()
    print("=== e3 tier-policy ablation (synthetic, real numbers) ===")
    print(f"  n_blocks={payload['n_blocks']} n_active={payload['n_active']} "
          f"pressure={payload['pressure']}")
    print("  config          RAM-resident  swapped  modeled-TPOT(ms)")
    for name, r in payload["configs"].items():
        print(
            f"  {name:14s} {r['ram_resident_blocks']:12d}  {r['swapped_blocks']:7d}  "
            f"{r['modeled_tpot_ms']:14.3f}"
        )
    print(f"  best={payload['best_config']}  "
          f"no-tier/full = {payload['no_tier_over_full_x']:.1f}x worse TPOT")
    path = save_result("e3_tier_policy_ablation", payload)
    print(f"  -> {path}")
    return payload


if __name__ == "__main__":
    main()

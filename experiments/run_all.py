"""Run the full UMA-LLM experiment suite and print a summary table.

Executes each ``eN_*`` driver's ``main()`` in order, collects the JSON each
writes under ``experiments/results/``, and prints a one-line-per-experiment
summary distinguishing synthetic (real-now) results from
hardware-placeholder ones.

Usage::

    python experiments/run_all.py

The synthetic drivers (e2 sizing frontier, e3 tier-policy ablation, e6
precision sweep) yield real numbers in the sandbox; the rest emit
``_is_measured=False`` placeholder rows off-hardware.
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

import importlib  # noqa: E402
import traceback  # noqa: E402

from . import DRIVERS, SYNTHETIC_DRIVERS, add_repo_to_path  # noqa: E402

add_repo_to_path()


def run_all() -> dict:
    """Run every driver, returning ``{name: payload | {"_error": ...}}``."""
    collected: dict[str, dict] = {}
    for name in DRIVERS:
        mod = importlib.import_module(f"experiments.{name}")
        print(f"\n----- running {name} -----")
        try:
            payload = mod.main()
        except Exception:  # keep going so one failure doesn't hide the rest
            traceback.print_exc()
            collected[name] = {"_error": traceback.format_exc().splitlines()[-1]}
            continue
        collected[name] = payload
    return collected


def _summary_line(name: str, payload: dict) -> str:
    if "_error" in payload:
        return f"  {name:28s} ERROR: {payload['_error']}"
    measured = payload.get("_is_measured")
    synthetic = name in SYNTHETIC_DRIVERS
    # e4 is also a closed-form / modeled result (real now, but not hardware
    # measured), so label it "modeled" rather than "measured".
    if synthetic:
        kind = "synthetic(real)"
    elif name == "e4_pressure_response":
        kind = "modeled(real)"
    elif measured:
        kind = "measured"
    else:
        kind = "placeholder"
    extra = ""
    if name == "e2_sizing_frontier":
        extra = (f"actionable={payload['n_actionable']}/{payload['n_cells']} "
                 f"floor={payload['deadline_floor_ms']:.0f}ms")
    elif name == "e3_tier_policy_ablation":
        extra = (f"full={payload['full_tpot_ms']:.2f}ms "
                 f"no_tier={payload['no_tier_tpot_ms']:.2f}ms "
                 f"({payload['no_tier_over_full_x']:.1f}x)")
    elif name == "e6_precision_sweep":
        m = payload["methods"]
        extra = (f"4bit_rmse={m['kivi_4bit']['relative_rmse']:.3f} "
                 f"2bit_rmse={m['kivi_2bit']['relative_rmse']:.3f}")
    elif name == "e1_cost_model_validation":
        mx = payload.get("max_rel_error")
        extra = f"max_rel_err={mx*100:.1f}%" if mx is not None else ""
    elif name == "e4_pressure_response":
        extra = (f"ctrl/listener P99 @CRIT="
                 f"{payload['control_over_listener_p99_x']:.1f}x")
    elif name == "e5_cross_device":
        extra = f"devices={payload['n_devices']}"
    elif name in ("e7_e2e_enablement", "e8_llama_cpp_baseline"):
        extra = f"placeholder_rows={payload['n_placeholder_rows']}/{payload['n_rows']}"
    elif name == "e9_e2e_uma_decode":
        d = payload["tiered_4bit"]
        extra = (f"decoded={payload['config']['tokens_decoded']}tok "
                 f"4bit cos={d['fidelity']['mean_cosine']:.3f} "
                 f"{d['compression_x']:.2f}x")
    return f"  {name:28s} [{kind:15s}] {extra}"


def main() -> dict:
    collected = run_all()
    print("\n" + "=" * 72)
    print("UMA-LLM experiment suite summary")
    print("=" * 72)
    for name in DRIVERS:
        print(_summary_line(name, collected.get(name, {"_error": "not run"})))
    print("=" * 72)
    synth_ok = all(
        "_error" not in collected.get(n, {"_error": "x"}) for n in SYNTHETIC_DRIVERS
    )
    print(f"synthetic drivers (e2/e3/e6) produced real numbers: {synth_ok}")
    n_err = sum(1 for p in collected.values() if "_error" in p)
    print(f"errors: {n_err}/{len(DRIVERS)}")
    return collected


if __name__ == "__main__":
    main()

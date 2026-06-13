"""UMA-LLM experiment suite.

Each ``eN_*.py`` driver in this package is runnable standalone as
``python experiments/eN_*.py`` and writes a JSON result into
``experiments/results/<name>.json``.

The whole suite is **sandbox-safe**: every driver runs end-to-end on a
CPU-only machine using NumPy and the calibration probes / baseline
placeholders in :mod:`umallm`. Drivers that depend on real hardware
(MLX / Metal / GH200) emit rows tagged ``_is_measured=False`` so a later
run on the target device overwrites the placeholders without touching the
driver code. The fully synthetic drivers (``e2``, ``e3``, ``e6``) produce
real numbers immediately.

Shared helpers (``results_dir``, ``save_result``, ``RESULTS_DIR``,
``add_repo_to_path``) live here so each driver stays short and the JSON
layout stays uniform.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

# experiments/ -> repo root (next3/)
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EXPERIMENTS_DIR)
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

# The experiment driver names, in run order, for run_all.py.
DRIVERS = [
    "e1_cost_model_validation",
    "e2_sizing_frontier",
    "e3_tier_policy_ablation",
    "e4_pressure_response",
    "e5_cross_device",
    "e6_precision_sweep",
    "e7_e2e_enablement",
    "e8_llama_cpp_baseline",
    "e9_e2e_uma_decode",
]

# Synthetic/CPU drivers that must yield real (not placeholder) numbers in-sandbox.
SYNTHETIC_DRIVERS = [
    "e2_sizing_frontier", "e3_tier_policy_ablation", "e6_precision_sweep",
    "e9_e2e_uma_decode",
]


def add_repo_to_path() -> None:
    """Make ``import umallm`` work when a driver is run as a script.

    ``python experiments/e2_sizing_frontier.py`` would otherwise only have
    ``experiments/`` on ``sys.path``; we prepend the repo root so the
    package resolves without requiring ``PYTHONPATH`` to be set by hand.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


def results_dir() -> str:
    """Return the results directory, creating it if necessary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def save_result(name: str, payload: dict) -> str:
    """Serialise ``payload`` to ``results/<name>.json`` and return the path.

    A ``_generated_at`` (UTC ISO timestamp) and ``_experiment`` key are
    injected so a later real-hardware run is distinguishable from the
    sandbox placeholder run.
    """
    out = dict(payload)
    out.setdefault("_experiment", name)
    out["_generated_at"] = datetime.now(timezone.utc).isoformat()
    path = os.path.join(results_dir(), f"{name}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    return path


def _json_default(obj):
    """Fallback JSON encoder for NumPy scalars / arrays."""
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")

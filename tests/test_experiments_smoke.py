"""Smoke tests for the fully-synthetic experiment drivers (e2/e3/e6).

These three drivers must produce real numbers in the sandbox, so we import
and run them in a tiny configuration and assert they return a populated
results dict and that ``main()`` writes the JSON file. The hardware-bound
drivers (e1/e4/e5/e7/e8) are exercised via ``run_all`` elsewhere; here we
keep the smoke test fast and synthetic-only per the task spec.
"""
import json
import os

import experiments
from experiments import e2_sizing_frontier, e3_tier_policy_ablation, e6_precision_sweep


def test_e2_run_produces_actionable_cells():
    payload = e2_sizing_frontier.run(
        n_blocks=64, deadlines_ms=[25, 50, 100], miss_targets=[1e-2]
    )
    assert payload["_is_measured"] is True
    assert payload["n_cells"] == 6  # 1 mode-pair x ... = 3 deadlines x 2 modes x 1 rho
    assert payload["n_actionable"] >= 1
    # Deadline floor is the 25 ms compute+attention sum.
    assert abs(payload["deadline_floor_ms"] - 25.0) < 1e-6
    assert payload["monotone_in_deadline"] is True
    assert payload["bernstein_at_least_as_conservative"] is True


def test_e3_run_tiered_beats_no_tier():
    payload = e3_tier_policy_ablation.run(n_blocks=64, n_active=24)
    assert payload["_is_measured"] is True
    assert set(payload["configs"]) == {"full", "compress_only", "no_pressure", "no_tier"}
    # no-tier pays the swap penalty -> strictly worse modeled TPOT than full.
    assert payload["no_tier_over_full_x"] > 1.0
    for cfg in payload["configs"].values():
        assert cfg["modeled_tpot_us"] > 0


def test_e6_run_reports_quant_error():
    payload = e6_precision_sweep.run(n_tokens=64, d_head=16, n_blocks=2)
    assert payload["_is_measured"] is True
    assert "kivi_4bit" in payload["methods"] and "kivi_2bit" in payload["methods"]
    # 4-bit relative RMSE is small and positive; ~4x compression vs fp16.
    m4 = payload["methods"]["kivi_4bit"]
    assert 0.0 < m4["relative_rmse"] < 0.5
    assert m4["compression_ratio_vs_fp16"] > 2.5
    # The driver self-reports whether the 2-bit number is plausible.
    assert "rmse_2bit_plausible" in payload


def test_payload_is_json_serialisable(tmp_path):
    """The e2 payload (with NumPy scalars / sets) round-trips through the
    package's JSON encoder into a tmp file with the injected metadata."""
    payload = e2_sizing_frontier.run(deadlines_ms=[50, 100], miss_targets=[1e-2])
    payload["_experiment"] = "e2_smoke"
    p = tmp_path / "e2_smoke.json"
    p.write_text(json.dumps(payload, indent=2, default=experiments._json_default))
    loaded = json.loads(p.read_text())
    assert loaded["_experiment"] == "e2_smoke"
    assert "cells" in loaded and len(loaded["cells"]) == 4


def test_e2_main_returns_and_writes():
    """The driver entry point returns a payload and writes its JSON file.

    ``main()`` writes to the canonical results dir (overwritten on every
    run, so no cleanup needed); we assert the file exists and is loadable.
    """
    payload = e2_sizing_frontier.main()
    assert payload["n_actionable"] >= 1
    expected = os.path.join(experiments.RESULTS_DIR, "e2_sizing_frontier.json")
    assert os.path.exists(expected)
    loaded = json.loads(open(expected).read())
    assert loaded["_experiment"] == "e2_sizing_frontier"
    assert "_generated_at" in loaded

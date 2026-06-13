"""CPU tests for the Route B benchmark harness via the MockBackend.

These validate the orchestration + metric pipeline AND that the mock's cost
model encodes the paper's hypothesis (the ordering Route B is expected to win).
The mock is synthetic; these are not hardware claims.
"""
import json

import pytest

from experiments.routeb_benchmark import (
    MODES, BenchConfig, MockBackend, build_workload, make_backend, main,
    run_matrix,
)


def _rows(results, mode):
    return [r for r in results if r["mode"] == mode]


def test_make_backend_mock_is_unmeasured():
    b = make_backend("mock")
    assert b.measured is False and b.name == "mock"
    with pytest.raises(ValueError):
        make_backend("nonsense")


def test_run_matrix_one_row_per_mode_per_config():
    cfgs = [BenchConfig(context_tokens=4096, n_requests=8, concurrency=8),
            BenchConfig(context_tokens=65536, n_requests=8, concurrency=8)]
    res = run_matrix(MODES, cfgs, MockBackend())
    assert len(res) == len(MODES) * len(cfgs)
    assert all(r["measured"] is False for r in res)
    assert {r["mode"] for r in res} == set(MODES)


def test_low_context_everything_fits_no_failures():
    # 4096 tokens * 8 reqs of 8B KV is well under a 40GiB HBM budget
    cfg = BenchConfig(context_tokens=4096, n_requests=16, concurrency=16,
                      hbm_budget_gib=40.0)
    res = run_matrix(MODES, [cfg], MockBackend())
    for r in res:
        assert r["n_failed"] == 0, f"{r['mode']} should fit at low context"


def test_high_context_hbm_only_ooms_others_survive():
    # 65536 tokens * 16 reqs of 8B KV (~128KiB/token) >> 40GiB HBM budget
    cfg = BenchConfig(context_tokens=65536, n_requests=32, concurrency=16,
                      hbm_budget_gib=40.0)
    res = run_matrix(MODES, [cfg], MockBackend())
    hbm = _rows(res, "hbm_only")[0]
    assert hbm["n_failed"] > 0          # capacity ceiling hit -> OOM
    for mode in ("routeb", "passive_uvm", "vllm_offload"):
        assert _rows(res, mode)[0]["n_failed"] == 0  # Grace/host absorbs cold KV


def test_hypothesis_routeb_best_tail_passive_worst():
    cfg = BenchConfig(context_tokens=65536, n_requests=32, concurrency=16,
                      hbm_budget_gib=40.0, deadline_ms=50.0)
    res = run_matrix(("routeb", "passive_uvm", "vllm_offload"), [cfg], MockBackend())
    rb = _rows(res, "routeb")[0]
    pu = _rows(res, "passive_uvm")[0]
    vo = _rows(res, "vllm_offload")[0]
    # Route B has the best (lowest) tail; passive UVM the worst (fault thrash).
    assert rb["tpot_p99_ms"] < pu["tpot_p99_ms"]
    assert rb["tpot_p99_ms"] <= vo["tpot_p99_ms"] + 1.0
    assert vo["tpot_p99_ms"] < pu["tpot_p99_ms"]
    # ...and the best goodput under the SLO
    assert rb["goodput_tok_s"] >= vo["goodput_tok_s"]
    assert rb["goodput_tok_s"] >= pu["goodput_tok_s"]


def test_routeb_records_grace_residency_when_cold():
    cfg = BenchConfig(context_tokens=65536, n_requests=16, concurrency=16,
                      hbm_budget_gib=40.0)
    res = run_matrix(("routeb",), [cfg], MockBackend())[0]
    assert res["cold_frac"] > 0.0
    assert res["grace_resident_gib"] > 0.0
    assert res["hbm_high_water_gib"] <= cfg.hbm_budget_gib + 1e-6


def test_multi_turn_workload_runs():
    cfg = BenchConfig(workload="multi_turn", context_tokens=4096,
                      n_requests=16, concurrency=8)
    specs = build_workload(cfg)
    assert any(s.conversation_id for s in specs)
    res = run_matrix(("routeb",), [cfg], MockBackend())
    assert res[0]["n_ok"] > 0


def test_kv_bytes_per_token_geometry():
    cfg = BenchConfig(n_layers=32, n_kv_heads=8, head_dim=128, dtype_bytes=2)
    # 2 * 32 * 8 * 128 * 2 = 131072 bytes/token = 128 KiB/token
    assert cfg.kv_bytes_per_token() == 131072


def test_main_writes_json(tmp_path):
    out = tmp_path / "mock.json"
    rc = main(["--backend", "mock", "--context-sweep", "4096,65536",
               "--n-requests", "8", "--concurrency", "8", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["measured"] is False
    assert len(payload["results"]) == len(MODES) * 2

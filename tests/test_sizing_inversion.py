"""Tests for the schedulability inversion ``min_active_blocks_for_slo``.

Covers the four structural properties an ICCD reviewer would check:
  * monotonicity: a looser deadline never needs a larger active budget;
  * infeasibility below the deadline floor;
  * the achieved tail bound is <= the miss target when feasible;
  * Bernstein is never less conservative than sub-Gaussian.
Plus a guard that the legacy ``min_budget_for_slo`` still returns a
positive int.
"""
import math

from umallm.uma_model import SizingResult, UMACostModel

# A representative single-request working set: 25 ms compute+attention floor.
EMA_ATTN = 10_000.0
EMA_COMPUTE = 15_000.0
N_BLOCKS = 512
FLOOR_US = EMA_ATTN + EMA_COMPUTE  # 25 ms


def _solve(model, deadline_us, miss_target=1e-2, mode="subgaussian"):
    return model.min_active_blocks_for_slo(
        deadline_us=deadline_us,
        ema_attention_lat=EMA_ATTN,
        ema_compute_lat=EMA_COMPUTE,
        n_blocks=N_BLOCKS,
        miss_target=miss_target,
        mode=mode,
    )


def test_returns_sizing_result():
    m = UMACostModel()
    res = _solve(m, 50_000.0)
    assert isinstance(res, SizingResult)
    assert 1 <= res.min_active_blocks <= N_BLOCKS
    assert 0.0 <= res.max_slow_fraction <= 1.0
    assert math.isclose(res.deadline_floor_us, FLOOR_US, rel_tol=1e-6)


def test_monotone_looser_deadline_smaller_budget():
    """Looser deadline => min_active is non-increasing."""
    m = UMACostModel()
    deadlines = [30_000.0, 35_000.0, 50_000.0, 75_000.0, 100_000.0, 200_000.0]
    budgets = [_solve(m, d).min_active_blocks for d in deadlines]
    for a, b in zip(budgets, budgets[1:]):
        assert b <= a, f"budget increased with looser deadline: {budgets}"
    # The sweep must actually move (not a degenerate all-equal sequence).
    assert budgets[0] > budgets[-1]


def test_infeasible_below_deadline_floor():
    """Deadline at or below the compute+attention floor is infeasible."""
    m = UMACostModel()
    res = _solve(m, FLOOR_US)  # exactly on the floor
    assert res.feasible is False
    assert res.min_active_blocks == N_BLOCKS
    assert res.max_slow_fraction == 0.0
    # And clearly-below-floor is infeasible too.
    res2 = _solve(m, FLOOR_US * 0.5)
    assert res2.feasible is False


def test_bound_at_solution_within_target_when_feasible():
    m = UMACostModel()
    for rho in (1e-2, 1e-3):
        for d in (35_000.0, 50_000.0, 100_000.0):
            res = _solve(m, d, miss_target=rho)
            if res.feasible:
                assert res.bound_at_solution <= rho + 1e-12, (
                    f"bound {res.bound_at_solution} > rho {rho} at D={d}"
                )


def test_bernstein_at_least_as_conservative():
    """Bernstein budget >= sub-Gaussian budget at matched (D, rho)."""
    m = UMACostModel()
    any_strictly_larger = False
    for d in (35_000.0, 45_000.0, 50_000.0, 60_000.0):
        sg = _solve(m, d, mode="subgaussian")
        bn = _solve(m, d, mode="bernstein")
        assert bn.min_active_blocks >= sg.min_active_blocks
        if bn.min_active_blocks > sg.min_active_blocks:
            any_strictly_larger = True
    # On at least one mid-range deadline Bernstein should be strictly more
    # conservative (otherwise the two modes are indistinguishable here).
    assert any_strictly_larger


def test_higher_predictor_recall_admits_smaller_budget():
    """A better hot-set predictor (recall) should never need more blocks."""
    m = UMACostModel()
    lo = m.min_active_blocks_for_slo(
        deadline_us=40_000.0, ema_attention_lat=EMA_ATTN,
        ema_compute_lat=EMA_COMPUTE, n_blocks=N_BLOCKS, predictor_recall=0.0,
    )
    hi = m.min_active_blocks_for_slo(
        deadline_us=40_000.0, ema_attention_lat=EMA_ATTN,
        ema_compute_lat=EMA_COMPUTE, n_blocks=N_BLOCKS, predictor_recall=0.5,
    )
    assert hi.min_active_blocks <= lo.min_active_blocks


def test_legacy_min_budget_for_slo_positive_int():
    m = UMACostModel()
    n = m.min_budget_for_slo(
        deadline_us=50_000.0, ema_attention_lat=EMA_ATTN,
        ema_compute_lat=EMA_COMPUTE, n_blocks=N_BLOCKS, slow_tier_fraction=0.10,
    )
    assert isinstance(n, int)
    assert n > 0
    # The sub-exponential legacy flag must also return a positive int.
    n2 = m.min_budget_for_slo_subgaussian(
        deadline_us=50_000.0, ema_attention_lat=EMA_ATTN,
        ema_compute_lat=EMA_COMPUTE, n_blocks=N_BLOCKS, use_sub_exponential=True,
    )
    assert isinstance(n2, int) and n2 > 0

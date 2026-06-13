"""Tests for the UMA cost model."""
import numpy as np
import pytest

from umallm.uma_model import ResidencyTier, UMACostModel


def test_same_tier_zero_cost():
    m = UMACostModel()
    for t in ResidencyTier:
        assert m.cost(t, t) == 0.0


def test_t0_t1_cheap():
    """T0 ↔ T1 should be just cache-warm cost (no copy)."""
    m = UMACostModel()
    c = m.cost(ResidencyTier.T0_GPU_ACTIVE, ResidencyTier.T1_CPU_ACTIVE)
    # 32 KB / 128 B per line = 256 lines × 80 ns = 20.48 µs
    assert 15 < c < 25


def test_t2_compression_cost():
    m = UMACostModel(compress_us_per_kb=1.5)
    c = m.cost(ResidencyTier.T0_GPU_ACTIVE, ResidencyTier.T2_COMPRESSED)
    # 32 KB × 1.5 µs/KB = 48 µs
    assert abs(c - 48.0) < 1.0


def test_t3_swap_round_trip_dominates():
    m = UMACostModel()
    in_ = m.cost(ResidencyTier.T3_SWAPPED, ResidencyTier.T0_GPU_ACTIVE)
    out_ = m.cost(ResidencyTier.T0_GPU_ACTIVE, ResidencyTier.T3_SWAPPED)
    # swap-in (10 µs/KB × 32) >> compression (1.5 × 32)
    assert in_ > 200
    assert out_ > 50


def test_access_cost_ordering():
    """T0 < T1 < T2 < T3 in access latency."""
    m = UMACostModel()
    c = [m.access_cost(t) for t in [
        ResidencyTier.T0_GPU_ACTIVE,
        ResidencyTier.T1_CPU_ACTIVE,
        ResidencyTier.T2_COMPRESSED,
        ResidencyTier.T3_SWAPPED,
    ]]
    assert c[0] < c[1] < c[2] < c[3], f"unexpected ordering: {c}"


def test_min_budget_for_slo_returns_int():
    m = UMACostModel()
    n = m.min_budget_for_slo(
        deadline_us=50_000.0, ema_attention_lat=10_000.0,
        ema_compute_lat=15_000.0, n_blocks=512, slow_tier_fraction=0.10,
    )
    assert isinstance(n, int)
    assert n > 0

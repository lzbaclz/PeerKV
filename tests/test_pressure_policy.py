"""Tests for pressure listener and UMA policy."""
import numpy as np

from umallm.policy import UMAPolicy
from umallm.pressure import MockPressureListener, PressureLevel
from umallm.uma_model import ResidencyTier, UMACostModel


def test_policy_default_pressure_distribution():
    p = UMAPolicy(n_active=10, gpu_frac=0.7, n_sink=2, n_window=2)
    n = 20
    scores = np.linspace(0.1, 0.9, n)
    tiers = p.place(scores, pressure=PressureLevel.NORMAL)
    # sink + window forced to T0
    for i in [0, 1, n - 2, n - 1]:
        assert tiers[i] == int(ResidencyTier.T0_GPU_ACTIVE)
    # T0 + T1 count == n_active (or less if budget overlap)
    n_active = ((tiers == int(ResidencyTier.T0_GPU_ACTIVE)) |
                (tiers == int(ResidencyTier.T1_CPU_ACTIVE))).sum()
    assert n_active <= 10


def test_policy_warn_compresses_more():
    p = UMAPolicy(n_active=12, gpu_frac=0.5, n_sink=0, n_window=0)
    n = 30
    scores = np.linspace(0.1, 0.9, n)
    t_norm = p.place(scores, pressure=PressureLevel.NORMAL)
    p2 = UMAPolicy(n_active=12, gpu_frac=0.5, n_sink=0, n_window=0)
    t_warn = p2.place(scores, pressure=PressureLevel.WARN)
    n_t2_norm = (t_norm == int(ResidencyTier.T2_COMPRESSED)).sum()
    n_t2_warn = (t_warn == int(ResidencyTier.T2_COMPRESSED)).sum()
    assert n_t2_warn >= n_t2_norm


def test_policy_critical_introduces_swap():
    p = UMAPolicy(n_active=4, gpu_frac=0.5, n_sink=0, n_window=0)
    n = 40
    scores = np.linspace(0.1, 0.9, n)
    t_crit = p.place(scores, pressure=PressureLevel.CRITICAL)
    n_t3 = (t_crit == int(ResidencyTier.T3_SWAPPED)).sum()
    assert n_t3 > 0


def test_mock_pressure_listener():
    received = []
    listener = MockPressureListener(callback=lambda lvl: received.append(lvl))
    listener.inject(PressureLevel.WARN)
    listener.inject(PressureLevel.CRITICAL)
    listener.inject(PressureLevel.NORMAL)
    assert received == [PressureLevel.WARN, PressureLevel.CRITICAL, PressureLevel.NORMAL]


def test_mock_pressure_dedup():
    received = []
    listener = MockPressureListener(callback=lambda lvl: received.append(lvl))
    listener.inject(PressureLevel.WARN)
    listener.inject(PressureLevel.WARN)
    listener.inject(PressureLevel.WARN)
    # only one notification: same level
    assert received == [PressureLevel.WARN]

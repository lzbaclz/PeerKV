"""Tests for the 5 SOTA-driven iterations on UMA-LLM."""
import numpy as np

from umallm.compression import (
    quantize_block,
    quantize_block_2bit,
    dequantize_block,
    compression_error,
)
from umallm.grace_hopper import GraceHopperCostModel
from umallm.head_profile import HeadPriorityProfile, head_aware_score
from umallm.sparse_predict import NextStepHotPredictor
from umallm.uma_model import ResidencyTier, UMACostModel


# Iter 1 — Sparse predictor
def test_sparse_predictor_returns_unit_interval():
    p = NextStepHotPredictor()
    B = 32
    rng = np.random.default_rng(0)
    out = p.predict(
        ema_attention=rng.random(B).astype(np.float32),
        last_used=rng.integers(0, 100, size=B).astype(np.float32),
        step=100,
    )
    assert out.shape == (B,)
    assert (out >= 0).all() and (out <= 1).all()


def test_select_for_promotion_returns_top_k():
    p = NextStepHotPredictor()
    B = 32
    rng = np.random.default_rng(0)
    idx = p.select_for_promotion(
        ema_attention=rng.random(B).astype(np.float32),
        last_used=np.zeros(B, dtype=np.float32),
        step=10, top_k=5,
    )
    assert idx.shape == (5,)
    assert len(set(idx.tolist())) == 5  # unique


# Iter 2 — Head profile
def test_head_priority_from_traces():
    rng = np.random.default_rng(0)
    # synthetic: 4 traces of (4 layers, 8 heads, 16 positions)
    traces = [rng.random((4, 8, 16)).astype(np.float32) for _ in range(4)]
    p = HeadPriorityProfile.from_attention_traces(traces)
    assert p.priorities.shape == (4, 8)
    assert (p.priorities >= 0).all() and (p.priorities <= 1).all()


def test_head_priority_save_load(tmp_path):
    p = HeadPriorityProfile(priorities=np.array([[0.1, 0.9], [0.5, 0.5]]).astype(np.float32))
    out = tmp_path / "prof.json"
    p.save(out)
    loaded = HeadPriorityProfile.load(out)
    np.testing.assert_allclose(loaded.priorities, p.priorities)


def test_head_aware_score():
    base = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    priors = np.array([0.5, 1.0], dtype=np.float32)
    assignment = np.array([0, 1, 0, 1], dtype=np.int64)
    out = head_aware_score(base, priors, assignment)
    np.testing.assert_allclose(out, [0.5, 1.0, 0.5, 1.0])


# Iter 3 — 2-bit quantization
def test_quantize_2bit_packs_four_per_byte():
    K = np.random.default_rng(0).normal(size=(32, 128)).astype(np.float32)
    c4 = quantize_block(K)
    c2 = quantize_block_2bit(K)
    # 2-bit packs 4 values/byte vs the 4-bit path's 2/byte -> ~half the bytes.
    assert c2.bits == 2
    assert c2.packed.shape[0] <= (c4.packed.shape[0] // 2) + 1


def test_quantize_2bit_roundtrips_with_higher_but_bounded_error():
    K = np.random.default_rng(0).normal(size=(32, 128)).astype(np.float32)
    c4 = quantize_block(K)
    c2 = quantize_block_2bit(K)
    e4 = compression_error(K, c4)["relative_rmse"]
    e2 = compression_error(K, c2)["relative_rmse"]
    # 2-bit is lossier than 4-bit, but the round-trip must be sane now
    # (a small multiple, not the ~90x the old broken dequant produced).
    assert e2 > e4
    assert e2 < 1.0


# Iter 5 — Grace-Hopper
def test_gh200_t0_t1_equal_cost():
    """On GH200, T0 (GPU) and T1 (CPU) have same access cost (NVLink-C2C)."""
    m = GraceHopperCostModel()
    c_t0 = m.access_cost(ResidencyTier.T0_GPU_ACTIVE)
    c_t1 = m.access_cost(ResidencyTier.T1_CPU_ACTIVE)
    assert c_t0 == c_t1


def test_gh200_higher_bandwidth_than_m2_max():
    gh = GraceHopperCostModel()
    m2 = UMACostModel()
    assert gh.soc_bw_gbps > m2.soc_bw_gbps
    # access cost should be lower on GH200
    assert gh.access_cost(ResidencyTier.T0_GPU_ACTIVE) < m2.access_cost(ResidencyTier.T0_GPU_ACTIVE)

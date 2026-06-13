"""End-to-end runtime tests: a model actually decodes through the tiered cache."""
import numpy as np
import pytest

from umallm.backends.numpy_ref import RefTransformer
from umallm.kv_runtime import (
    TieredKVCache, greedy_decode, tiered_score_sequence, logit_fidelity,
)
from umallm.policy import UMAPolicy


def _cache(model, cold_bits=4, compress=True, n_active=3, block_size=8):
    pol = UMAPolicy(n_active=n_active, n_sink=1, n_window=1)
    return TieredKVCache(model.n_layers, model.n_heads, model.head_dim,
                         block_size=block_size, policy=pol,
                         cold_bits=cold_bits, compress=compress)


def test_e2e_decode_runs_and_returns_valid_tokens():
    m = RefTransformer(n_layers=2, n_heads=4, head_dim=16, vocab=64, seed=1)
    prompt = list(range(20))
    gen = greedy_decode(m, prompt, n_gen=8, cache=_cache(m))
    assert len(gen) == 8
    assert all(0 <= t < m.vocab for t in gen)


def test_full_cache_is_exact_identity():
    """compress=False must reproduce itself bit-for-bit (sanity)."""
    m = RefTransformer(seed=2)
    seq = list(np.random.default_rng(2).integers(0, m.vocab, size=40))
    a = tiered_score_sequence(m, seq, _cache(m, compress=False))
    b = tiered_score_sequence(m, seq, _cache(m, compress=False))
    assert np.allclose(a, b)
    assert logit_fidelity(a, b)["rel_l2"] == pytest.approx(0.0, abs=1e-6)


def test_compression_reduces_kv_bytes():
    m = RefTransformer(seed=3)
    seq = list(np.random.default_rng(3).integers(0, m.vocab, size=64))
    c = _cache(m, cold_bits=4, compress=True)
    tiered_score_sequence(m, seq, c)
    fp = c.footprint()
    assert fp["actual_bytes"] < fp["fp16_bytes"]      # tiering saved memory
    assert fp["compression_x"] > 1.0
    assert fp["tier_block_counts"][2] > 0             # some blocks compressed


def test_4bit_fidelity_high_and_2bit_lossier():
    m = RefTransformer(seed=4)
    seq = list(np.random.default_rng(4).integers(0, m.vocab, size=64))
    full = tiered_score_sequence(m, seq, _cache(m, compress=False))
    t4 = tiered_score_sequence(m, seq, _cache(m, cold_bits=4, compress=True))
    t2 = tiered_score_sequence(m, seq, _cache(m, cold_bits=2, compress=True))
    f4 = logit_fidelity(full, t4)
    f2 = logit_fidelity(full, t2)
    assert f4["mean_cosine"] > 0.99          # 4-bit barely perturbs the output
    assert f4["rel_l2"] < 0.1
    assert f2["rel_l2"] > f4["rel_l2"]        # 2-bit is lossier end-to-end


def test_more_context_triggers_more_compression():
    m = RefTransformer(seed=5)
    rng = np.random.default_rng(5)
    short = list(rng.integers(0, m.vocab, size=32))
    long = list(rng.integers(0, m.vocab, size=96))
    cs, cl = _cache(m), _cache(m)
    tiered_score_sequence(m, short, cs)
    tiered_score_sequence(m, long, cl)
    assert cl.footprint()["tier_block_counts"][2] >= cs.footprint()["tier_block_counts"][2]

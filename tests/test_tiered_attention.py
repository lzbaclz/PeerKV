"""Correctness of MLXTieredCache.tiered_attention vs dense causal SDPA.

all-hot mode (nothing cold) must match dense attention EXACTLY (modulo fp16);
tiered mode (middle blocks low-bit) matches within quantization error. Covers
multi-chunk prefill (L>1) + decode (L=1), and MHA + GQA. Skips without MLX.
"""
import numpy as np
import pytest

from umallm.mlx_backend import HAS_MLX

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

if HAS_MLX:
    import mlx.core as mx
    from umallm.mlx_tiered_attention import MLXTieredCache

D = 64


def _cos(a, b):
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _ref(q, fk, fv, n_rep, scale):
    if n_rep > 1:
        fk = mx.repeat(fk, n_rep, axis=1)
        fv = mx.repeat(fv, n_rep, axis=1)
    L, T = q.shape[2], fk.shape[2]
    qpos = mx.arange(T - L, T).reshape(L, 1)
    kpos = mx.arange(T).reshape(1, T)
    add = mx.where(qpos >= kpos, mx.array(0.0, mx.float32),
                   mx.array(-1e9, mx.float32)).astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(q, fk, fv, scale=scale, mask=add)


def _drive(n_kv, n_q, all_hot, seed=0):
    rng = np.random.default_rng(seed)
    n_rep = n_q // n_kv
    scale = 1.0 / np.sqrt(D)
    cache = (MLXTieredCache(block_size=16, n_sink_blocks=0, n_window_blocks=10_000)
             if all_hot else
             MLXTieredCache(block_size=16, n_sink_blocks=1, n_window_blocks=1,
                            cold_bits=4, group_size=64))
    fk = fv = None
    worst_max, worst_cos = 0.0, 1.0
    for L in [24, 16] + [1] * 12:
        q = mx.array(rng.standard_normal((1, n_q, L, D)).astype(np.float32), mx.float16)
        k = mx.array(rng.standard_normal((1, n_kv, L, D)).astype(np.float32), mx.float16)
        v = mx.array(rng.standard_normal((1, n_kv, L, D)).astype(np.float32), mx.float16)
        fk = k if fk is None else mx.concatenate([fk, k], axis=2)
        fv = v if fv is None else mx.concatenate([fv, v], axis=2)
        cache.update_and_fetch(k, v)
        ot = cache.tiered_attention(q, scale); mx.eval(ot)
        orf = _ref(q, fk, fv, n_rep, scale); mx.eval(orf)
        worst_max = max(worst_max, float(np.max(np.abs(
            np.asarray(orf).astype(np.float64) - np.asarray(ot).astype(np.float64)))))
        worst_cos = min(worst_cos, _cos(orf, ot))
    return worst_max, worst_cos, cache.footprint()


@pytest.mark.parametrize("n_kv,n_q", [(4, 4), (4, 16)])
def test_all_hot_matches_dense_exactly(n_kv, n_q):
    worst_max, worst_cos, fp = _drive(n_kv, n_q, all_hot=True)
    assert fp["cold_tokens"] == 0
    assert worst_max < 5e-3, worst_max          # exact modulo fp16
    assert worst_cos > 0.999


@pytest.mark.parametrize("n_kv,n_q", [(4, 4), (4, 16)])
def test_tiered_matches_within_quant_error(n_kv, n_q):
    worst_max, worst_cos, fp = _drive(n_kv, n_q, all_hot=False)
    assert fp["cold_tokens"] > 0                # tiering actually happened
    assert worst_cos > 0.95, worst_cos          # only the 4-bit cold-tier error

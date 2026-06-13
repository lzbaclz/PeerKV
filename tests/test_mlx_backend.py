"""Tests for the MLX backend.

Most tests skip when MLX isn't available (sandbox/CI); the residency-tag
plumbing is tested without MLX.
"""
import numpy as np
import pytest

from umallm.mlx_backend import UMAKVCache, HAS_MLX
from umallm.uma_model import ResidencyTier


def test_uma_kv_cache_tier_tag_default():
    cache = UMAKVCache(num_layers=4, num_heads=2, head_dim=64,
                      max_blocks=16, block_size=32)
    stats = cache.residency_stats()
    # all default to T0
    assert stats[0] == 4 * 16


def test_uma_kv_cache_set_tier_no_compress():
    cache = UMAKVCache(num_layers=2, num_heads=2, head_dim=64, max_blocks=8)
    cache.set_tier(0, 0, ResidencyTier.T1_CPU_ACTIVE)
    assert int(cache.tiers[0, 0]) == int(ResidencyTier.T1_CPU_ACTIVE)


def test_uma_kv_cache_t0_to_t2_compresses():
    cache = UMAKVCache(num_layers=2, num_heads=2, head_dim=64, max_blocks=8)
    cache.set_tier(0, 1, ResidencyTier.T2_COMPRESSED)
    assert (0, 1) in cache._compressed


def test_uma_kv_cache_t2_to_t0_decompresses():
    cache = UMAKVCache(num_layers=2, num_heads=2, head_dim=64, max_blocks=8)
    cache.set_tier(0, 1, ResidencyTier.T2_COMPRESSED)
    cache.set_tier(0, 1, ResidencyTier.T0_GPU_ACTIVE)
    assert (0, 1) not in cache._compressed


def test_compressed_bytes_grows_with_promotion():
    cache = UMAKVCache(num_layers=1, num_heads=2, head_dim=64, max_blocks=8)
    b0 = cache.compressed_bytes()
    cache.set_tier(0, 0, ResidencyTier.T2_COMPRESSED)
    b1 = cache.compressed_bytes()
    assert b1 > b0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
def test_mlx_live_array_init():
    cache = UMAKVCache(num_layers=2, num_heads=2, head_dim=64, max_blocks=8)
    cache._ensure_live()
    assert cache._K is not None

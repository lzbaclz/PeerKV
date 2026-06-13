"""Tests for KIVI 4-bit quantize/dequantize."""
import numpy as np

from umallm.compression import (
    KIVI4bit,
    GROUP_SIZE,
    compression_error,
    dequantize_block,
    quantize_block,
)


def test_quantize_roundtrip_shape():
    K = np.random.default_rng(0).normal(size=(32, 128)).astype(np.float32)
    c = quantize_block(K)
    Kp = dequantize_block(c)
    assert Kp.shape == K.shape


def test_quantize_relative_rmse_bound():
    """KIVI 4-bit per-group with G=32 should give relative RMSE < 8%."""
    K = np.random.default_rng(0).normal(size=(32, 128)).astype(np.float32)
    c = quantize_block(K)
    err = compression_error(K, c)
    assert err["relative_rmse"] < 0.08, f"relative_rmse {err['relative_rmse']:.3f} too high"


def test_compression_ratio():
    K = np.random.default_rng(0).normal(size=(32, 128)).astype(np.float32)
    c = quantize_block(K)
    # original fp32 = 32*128*4 = 16384 bytes; fp16 = 8192 bytes
    # compressed: 4-bit + 2 fp16 per group (scale + zero)
    # 32*128*0.5 (4-bit) + 1*128*2 (scale, one group) + 1*128*2 (zero) = 2560 bytes
    # compression ratio vs fp16 ~3.2× (expected)
    ratio_fp16 = (K.astype(np.float16).nbytes) / c.nbytes()
    assert ratio_fp16 > 2.5, f"compression ratio {ratio_fp16:.2f} too low"


def test_quantize_handles_constant():
    """Edge case: constant block (zero variance in group)."""
    K = np.full((32, 128), 0.7, dtype=np.float32)
    c = quantize_block(K)
    Kp = dequantize_block(c)
    np.testing.assert_allclose(Kp, K, atol=0.01)


def test_quantize_packed_size():
    K = np.zeros((32, 128), dtype=np.float32)
    c = quantize_block(K)
    # 32 * 128 = 4096 elements; packed should be 4096/2 = 2048 bytes
    assert c.packed.shape == (2048,)
    assert c.packed.dtype == np.uint8


def test_multiple_groups():
    """N = 64, group_size=32 → 2 groups."""
    K = np.random.default_rng(0).normal(size=(64, 32)).astype(np.float32)
    c = quantize_block(K, group_size=32)
    assert c.scale.shape == (2, 32)
    Kp = dequantize_block(c, group_size=32)
    err = compression_error(K, c)
    assert err["relative_rmse"] < 0.10

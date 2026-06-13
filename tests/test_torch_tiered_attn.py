"""CPU validation of the block-streaming flash merge (umallm/torch_tiered_attn.py).

This is the *math* behind e18's NVLink-tiered attention: streaming KV block-by-
block with an online-softmax merge must equal dense attention over the full KV.
Runs on CPU (no GPU needed); the multi-GPU placement/peak/latency is e18 on A100.
"""
import pytest

from umallm.torch_tiered_attn import HAS_TORCH, flash_merge_attention

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


def _blocks(K, V, sizes):
    return list(K.split(sizes, dim=2)), list(V.split(sizes, dim=2))


def test_flash_merge_exact_vs_dense_fp32():
    torch.manual_seed(0)
    B, H, Lq, Tk, D = 1, 4, 1, 300, 64           # decode: Lq=1, no causal mask
    q = torch.randn(B, H, Lq, D, dtype=torch.float32)
    K = torch.randn(B, H, Tk, D, dtype=torch.float32)
    V = torch.randn(B, H, Tk, D, dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, K, V)     # dense (no mask)
    kb, vb = _blocks(K, V, [64, 64, 64, 64, 44])      # ragged blocks, sum=300
    out = flash_merge_attention(q, kb, vb)
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


def test_blockcount_invariant():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 1, 32, dtype=torch.float32)
    K = torch.randn(1, 2, 256, 32, dtype=torch.float32)
    V = torch.randn(1, 2, 256, 32, dtype=torch.float32)
    o1 = flash_merge_attention(q, *_blocks(K, V, [256]))      # 1 block
    o2 = flash_merge_attention(q, *_blocks(K, V, [32] * 8))   # 8 blocks
    o3 = flash_merge_attention(q, *_blocks(K, V, [100, 100, 56]))
    assert torch.allclose(o1, o2, atol=1e-5, rtol=1e-5)
    assert torch.allclose(o1, o3, atol=1e-5, rtol=1e-5)


def test_chunked_fetch_exact_vs_dense_and_perblock():
    # Coalescing consecutive blocks into one transfer (chunk_blocks>1) must not
    # change the result -- it is still exact flash merge, just bandwidth-bound.
    torch.manual_seed(7)
    B, H, Lq, Tk, D = 1, 4, 1, 320, 64
    q = torch.randn(B, H, Lq, D, dtype=torch.float32)
    K = torch.randn(B, H, Tk, D, dtype=torch.float32)
    V = torch.randn(B, H, Tk, D, dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, K, V)
    kb, vb = _blocks(K, V, [32] * 10)                 # 10 blocks of 32 tokens
    per_block = flash_merge_attention(q, kb, vb, chunk_blocks=1)
    for c in (2, 3, 5, 10, 100):                      # various chunk sizes incl. > nblocks
        out = flash_merge_attention(q, kb, vb, chunk_blocks=c)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
        assert torch.allclose(out, per_block, atol=1e-5, rtol=1e-5)


def test_multi_query_no_mask_matches_dense():
    torch.manual_seed(2)
    q = torch.randn(1, 4, 5, 64, dtype=torch.float32)    # Lq>1, no causal
    K = torch.randn(1, 4, 200, 64, dtype=torch.float32)
    V = torch.randn(1, 4, 200, 64, dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, K, V)
    out = flash_merge_attention(q, *_blocks(K, V, [50, 50, 50, 50]))
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


def test_fp16_close_to_dense():
    torch.manual_seed(3)
    q = torch.randn(1, 8, 1, 128, dtype=torch.float16)
    K = torch.randn(1, 8, 512, 128, dtype=torch.float16)
    V = torch.randn(1, 8, 512, 128, dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q, K, V).float().reshape(-1)
    out = flash_merge_attention(q, *_blocks(K, V, [128, 128, 128, 128])).float().reshape(-1)
    cos = float(torch.dot(ref, out) / (ref.norm() * out.norm() + 1e-9))
    assert cos > 0.99

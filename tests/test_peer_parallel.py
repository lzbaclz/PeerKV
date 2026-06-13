"""CPU validation of KV-parallel distributed attention (umallm/peer_parallel_attn.py).

The math behind e20: computing each KV shard's flash partial separately and merging
the (O, lse) statistics must equal dense attention over the concatenated KV --
regardless of how the KV is sharded across devices. Runs on CPU (no GPU needed); the
multi-GPU placement/latency is e20 on the dual-A100.
"""
import pytest

from umallm.peer_parallel_attn import (HAS_TORCH, flash_partial, merge_partial,
                                       peer_parallel_attention,
                                       ring_prefill_attention)

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


def _shards(K, V, sizes):
    ks = list(K.split(sizes, dim=2))
    vs = list(V.split(sizes, dim=2))
    return list(zip(ks, vs))


def test_two_shards_exact_vs_dense():
    torch.manual_seed(0)
    B, H, Lq, Tk, D = 1, 4, 1, 500, 64                # decode, no causal mask
    q = torch.randn(B, H, Lq, D)
    K = torch.randn(B, H, Tk, D)
    V = torch.randn(B, H, Tk, D)
    ref = F.scaled_dot_product_attention(q, K, V)
    out = peer_parallel_attention(q, _shards(K, V, [300, 200]))
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


def test_many_shards_and_order_invariant():
    torch.manual_seed(1)
    q = torch.randn(1, 8, 1, 128)
    K = torch.randn(1, 8, 1000, 128)
    V = torch.randn(1, 8, 1000, 128)
    ref = F.scaled_dot_product_attention(q, K, V)
    out = peer_parallel_attention(q, _shards(K, V, [128, 256, 384, 232]))
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)
    # merge is commutative/associative -> shard order must not change the result
    shards = _shards(K, V, [500, 500])
    o1 = peer_parallel_attention(q, shards)
    o2 = peer_parallel_attention(q, list(reversed(shards)))
    assert torch.allclose(o1, o2, atol=1e-5, rtol=1e-5)


def test_single_shard_equals_dense():
    torch.manual_seed(2)
    q = torch.randn(1, 2, 1, 32)
    K = torch.randn(1, 2, 137, 32)
    V = torch.randn(1, 2, 137, 32)
    ref = F.scaled_dot_product_attention(q, K, V)
    out = peer_parallel_attention(q, [(K, V)])
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


def test_multi_query_no_mask():
    torch.manual_seed(3)
    q = torch.randn(1, 4, 5, 64)                       # Lq>1, no causal
    K = torch.randn(1, 4, 200, 64)
    V = torch.randn(1, 4, 200, 64)
    ref = F.scaled_dot_product_attention(q, K, V)
    out = peer_parallel_attention(q, _shards(K, V, [50, 50, 100]))
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


def test_ring_prefill_causal_exact_vs_dense():
    torch.manual_seed(5)
    B, H, S, D = 1, 4, 600, 64
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    # split the sequence into 3 contiguous shards (ring order)
    cuts = [200, 200, 200]
    Qs = Q.split(cuts, dim=2); Ks = K.split(cuts, dim=2); Vs = V.split(cuts, dim=2)
    shards = list(zip(Qs, Ks, Vs))
    outs = ring_prefill_attention(shards)
    out = torch.cat(outs, dim=2)
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


def test_ring_prefill_single_shard_is_plain_causal():
    torch.manual_seed(6)
    Q = torch.randn(1, 2, 130, 48); K = torch.randn(1, 2, 130, 48)
    V = torch.randn(1, 2, 130, 48)
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    out = ring_prefill_attention([(Q, K, V)])[0]
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)


def test_fp16_long_seq_merge_stays_exact():
    # the sharded online-softmax merge must not accumulate error vs dense, even in
    # fp16 over a long sequence with many shards (GPU kernel path is checked at
    # scale by e20/e21/e25 with cos=1.000000; this bounds the merge math itself).
    torch.manual_seed(7)
    B, H, Lq, Tk, D = 1, 8, 1, 8192, 128
    q = torch.randn(B, H, Lq, D, dtype=torch.float16)
    K = torch.randn(B, H, Tk, D, dtype=torch.float16)
    V = torch.randn(B, H, Tk, D, dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q, K, V).float().reshape(-1)
    out = peer_parallel_attention(q, _shards(K, V, [512] * 16)).float().reshape(-1)
    cos = float(torch.dot(ref, out) / (ref.norm() * out.norm() + 1e-9))
    assert cos > 0.999
    assert (ref - out).abs().max() < 5e-2          # fp16 noise floor, no drift


def test_partial_then_merge_matches_dense():
    torch.manual_seed(4)
    q = torch.randn(1, 2, 1, 48)
    K = torch.randn(1, 2, 64, 48)
    V = torch.randn(1, 2, 64, 48)
    ref = F.scaled_dot_product_attention(q, K, V)
    (Ka, Va), (Kb, Vb) = _shards(K, V, [40, 24])
    Oa, la = flash_partial(q, Ka, Va)
    Ob, lb = flash_partial(q, Kb, Vb)
    O, _ = merge_partial(Oa, la, Ob, lb)
    assert torch.allclose(ref, O.to(q.dtype), atol=1e-4, rtol=1e-4)

"""Paged -> contiguous KV gather for the multi-device CFK attention op.

vLLM stores KV as paged blocks ``[2, num_blocks, block_size, H, D]`` indexed by a
per-sequence ``block_table``. The CFK op (fused_attn.peer_fused_attn) consumes
*contiguous* per-kv-head ``[HKV, T, D]`` shards. This module bridges the two for
decode: for each sequence it gathers the blocks that are resident locally (cuda:0)
into a contiguous local shard, and the blocks resident on the peer (cuda:1) into a
contiguous peer shard, then runs CFK and merges.

Why a set-split is exact: attention is permutation-invariant over keys, and the
online-softmax merge of two partials over *disjoint* key sets equals full attention.
So it is correct to put any subset of a sequence's tokens on the peer, in any order,
as long as every token is in exactly one shard.

Block residency convention (set by the allocator patch, peer_kv_alloc-style):
  global block id ``gid``:
    gid <  peer_base  -> local block, local_cache[*, gid]
    gid >= peer_base  -> peer  block, peer_cache[*, gid - peer_base]
Only the sequence's final block may be partially filled.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .fused_attn import peer_fused_attn


def _cat_blocks(cache: torch.Tensor, block_idx: list, valids: list,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """cache: [2, nb, bs, H, D]. Gather the listed blocks, trim to total valid tokens,
    return (K, V) as [H, T, D] contiguous on cache.device. Empty -> (T=0) tensors."""
    H, D = cache.size(3), cache.size(4)
    if not block_idx:
        z = cache.new_empty((H, 0, D))
        return z, z
    idx = torch.tensor(block_idx, device=cache.device, dtype=torch.long)
    K = cache[0].index_select(0, idx)   # [n, bs, H, D]
    V = cache[1].index_select(0, idx)
    n, bs = K.size(0), K.size(1)
    total_valid = sum(valids)           # partial block (if any) is last in this set
    K = K.reshape(n * bs, H, D)[:total_valid].permute(1, 0, 2).contiguous()  # [H, T, D]
    V = V.reshape(n * bs, H, D)[:total_valid].permute(1, 0, 2).contiguous()
    return K, V


def gather_seq(local_cache: torch.Tensor, peer_cache: Optional[torch.Tensor],
               block_row: torch.Tensor, seq_len: int, peer_base: int):
    """Split one sequence's blocks into contiguous (K_local,V_local) on cuda:0 and
    (K_peer,V_peer) on cuda:1. peer_cache/peer_base may be None/large to disable peer."""
    bs = local_cache.size(2)
    n_blk = (seq_len + bs - 1) // bs
    loc_idx, loc_val, peer_idx, peer_val = [], [], [], []
    for j in range(n_blk):
        gid = int(block_row[j].item())
        valid = bs if j < n_blk - 1 else (seq_len - (n_blk - 1) * bs)
        if peer_cache is None or gid < peer_base:
            loc_idx.append(gid); loc_val.append(valid)
        else:
            peer_idx.append(gid - peer_base); peer_val.append(valid)
    Kl, Vl = _cat_blocks(local_cache, loc_idx, loc_val)
    if peer_cache is None or not peer_idx:
        return Kl, Vl, None, None
    Kp, Vp = _cat_blocks(peer_cache, peer_idx, peer_val)
    return Kl, Vl, Kp, Vp


def paged_cfk_decode(query: torch.Tensor,
                     local_cache: torch.Tensor,
                     peer_cache: Optional[torch.Tensor],
                     block_table: torch.Tensor,
                     seq_lens: torch.Tensor,
                     peer_base: int,
                     scale: Optional[float] = None,
                     splits: int = 16) -> torch.Tensor:
    """Multi-sequence decode via per-sequence gather + CFK op.

    query:       [num_seqs, H, D]   (one decode token per sequence) on cuda:0
    local_cache: [2, nb_l, bs, H, D] on cuda:0
    peer_cache:  [2, nb_p, bs, H, D] on cuda:1 (or None for single-GPU)
    block_table: [num_seqs, max_blk]   global block ids
    seq_lens:    [num_seqs]
    returns:     [num_seqs, H, D] on cuda:0

    Correctness-first: loops sequences in Python (one op launch per seq). A batched
    kernel is a perf TODO; numerics are identical either way.
    """
    num_seqs, H, D = query.shape
    out = query.new_empty((num_seqs, H, D))
    for i in range(num_seqs):
        L = int(seq_lens[i].item())
        row = block_table[i]
        Kl, Vl, Kp, Vp = gather_seq(local_cache, peer_cache, row, L, peer_base)
        O = peer_fused_attn(query[i].half(), Kl, Vl, Kp, Vp, scale=scale, splits=splits)
        out[i] = O.to(query.dtype)
    return out

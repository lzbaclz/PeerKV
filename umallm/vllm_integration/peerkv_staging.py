"""PeerKV copy-back staging -- the engine-independent core of M2+M3.

The KV cache for a layer is split across two GPUs:
  * ``local_kv`` on cuda:0, shape (2, C_local + C_scratch, block_size, H, D).
    Blocks ``0 .. C_local-1`` are real local blocks; the last ``C_scratch``
    blocks are a reusable scratch region for staged-in peer blocks.
  * ``peer_kv``  on cuda:1, shape (2, C_peer, block_size, H, D).

Physical block-id convention (matches vLLM: it hands out ids 0..num_blocks-1):
  * ``0 .. C_local-1``      -> local block (kernel reads in place)
  * ``C_local .. C_local+C_peer-1`` -> peer block; peer index = id - C_local
  (scratch is the LOCAL tensor's tail, indices C_local..C_local+C_scratch-1; it is
   an internal remap target, never an input block-id, so no collision.)

Before the attention kernel runs we ``stage()``: for every peer block referenced
by this step's ``block_table`` we coalesce-copy it over NVLink into a scratch slot
of ``local_kv`` and rewrite the block_table entry to point at that scratch block.
The kernel then runs unmodified over a single cuda:0 tensor -- this is the
"compute reads local, link only carries the cold blocks once, coalesced" path
(copy-back), exactly the mechanism the paper's RQ2/RQ3 measure, now shaped for a
paged-attention engine.

This module is pure torch (no vLLM import) so it is unit-testable on any 2-GPU box
(see experiments/serve/m3_staging_test.py); the vLLM attention-backend wrapper that
calls it lives in peerkv_attn.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PeerKVLayout:
    c_local: int        # number of real local blocks on cuda:0
    c_scratch: int      # number of scratch blocks appended after the local blocks
    c_peer: int         # number of blocks resident on the peer GPU
    block_size: int
    num_kv_heads: int
    head_size: int
    local_device: str = "cuda:0"
    peer_device: str = "cuda:1"

    @property
    def peer_base(self) -> int:
        # vLLM's block manager hands out ids 0..num_blocks-1 (= c_local+c_peer-1)
        # with NO knowledge of tiering; we PHYSICALLY place ids >= c_local on the
        # peer GPU, so c_local is the peer threshold and peer index = id - c_local.
        # Scratch lives in the LOCAL tensor at indices c_local..c_local+c_scratch-1
        # and is an internal remap TARGET -- it never appears in an input
        # block_table, so there is no collision with the peer id range.
        return self.c_local


def alloc_split_kv(layout: PeerKVLayout, dtype=torch.float16):
    """Allocate the (local+scratch) and peer KV tensors. M2: this is the seam that
    vLLM's ``initialize_kv_cache`` (gpu_model_runner.py:1722) is wrapped to call so a
    fraction of physical blocks physically reside on the peer GPU."""
    local = torch.empty(2, layout.c_local + layout.c_scratch, layout.block_size,
                        layout.num_kv_heads, layout.head_size,
                        dtype=dtype, device=layout.local_device)
    peer = torch.empty(2, layout.c_peer, layout.block_size,
                       layout.num_kv_heads, layout.head_size,
                       dtype=dtype, device=layout.peer_device)
    return local, peer


class PeerKVStager:
    """Stages peer-resident blocks into ``local_kv`` scratch and remaps block ids.

    One stager per attention layer (or shared if scratch is sized for the batch).
    ``copy_stream`` overlaps the NVLink copy with prior compute when driven ahead.
    """

    def __init__(self, layout: PeerKVLayout):
        self.L = layout
        self._stream = torch.cuda.Stream(device=layout.local_device)

    @torch.no_grad()
    def stage(self, local_kv: torch.Tensor, peer_kv: torch.Tensor,
              block_table: torch.Tensor) -> torch.Tensor:
        """Copy every peer block referenced by ``block_table`` into ``local_kv``
        scratch (coalesced) and return a block_table whose peer ids now point at the
        scratch blocks. ``local_kv`` is mutated in place (scratch region only).

        block_table: int tensor [num_seqs, max_blocks_per_seq], ids per the
        convention above. Returns a new int tensor on ``local_kv.device``.
        """
        L = self.L
        bt = block_table.to(L.local_device)
        peer_mask = bt >= L.peer_base
        if not bool(peer_mask.any()):
            return bt  # nothing on the peer this step

        # unique peer blocks referenced -> assign contiguous scratch slots
        peer_ids = torch.unique(bt[peer_mask])                 # global ids
        peer_idx = (peer_ids - L.peer_base).to(torch.long)     # index into peer_kv
        n = peer_idx.numel()
        if n > L.c_scratch:
            raise RuntimeError(
                f"PeerKV: {n} peer blocks this step exceeds scratch capacity "
                f"{L.c_scratch}; raise c_scratch or shrink the peer working set.")
        scratch_slot = torch.arange(n, device=L.local_device)
        scratch_block = L.c_local + scratch_slot               # dst block id in local_kv

        # coalesced D2D copy peer_kv[peer_idx] -> local_kv[:, scratch_block]
        # (single gather+scatter; one NVLink stream, overlaps if driven ahead)
        # Ordering BOTH ways: the staging stream must wait for the current
        # stream before WRITING scratch (step N-1's attention kernel may still
        # be reading the same slots -- write-after-read hazard under async
        # serving), and the current stream must wait for the staging stream
        # before the kernel READS the new contents.
        cur = torch.cuda.current_stream(L.local_device)
        self._stream.wait_stream(cur)
        with torch.cuda.stream(self._stream):
            staged = peer_kv.index_select(1, peer_idx.to(peer_kv.device)).to(
                L.local_device, non_blocking=True)             # (2, n, bs, H, D)
            local_kv[:, scratch_block] = staged
        cur.wait_stream(self._stream)

        # remap: peer global id -> its scratch block id (vectorized via a lookup)
        remap = bt.clone()
        # build id->scratch map only over referenced peer ids
        # (small: n entries); scatter into a dense lut sized to max referenced id
        lut_size = int(peer_ids.max().item()) + 1
        lut = torch.full((lut_size,), -1, dtype=bt.dtype, device=L.local_device)
        lut[peer_ids] = scratch_block.to(bt.dtype)
        remap[peer_mask] = lut[bt[peer_mask]]
        return remap


@torch.no_grad()
def gather_kv(kv: torch.Tensor, block_table_row: torch.Tensor, n_blocks: int):
    """Test helper: gather the first ``n_blocks`` physical blocks named by a
    block_table row into a contiguous (2, n_blocks*block_size, H, D) tensor on
    kv.device. Used to prove staging is value-preserving."""
    ids = block_table_row[:n_blocks].to(kv.device).long()
    sel = kv.index_select(1, ids)                              # (2, n_blocks, bs, H, D)
    two, nb, bs, H, D = sel.shape
    return sel.reshape(two, nb * bs, H, D)

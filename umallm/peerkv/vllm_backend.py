"""Track C P3 entry: a vLLM custom attention impl that runs the CFK fused op.

Unlike the copy-back path (umallm/vllm_integration/peerkv_attn.py, which stages peer
blocks into local scratch then runs the stock paged kernel), this backend keeps the
peer KV shard ON the peer GPU and computes a peer attention partial there, merging it
with the local partial via the multi-device op in fused_attn.py. Only q (~KB) and the
(O,lse) partial (~KB) cross NVLink per layer; KV never moves.

Integration reality (honest, so nobody ships a broken default):
  * vLLM's paged attention reads KV from a paged block layout [2, num_blocks, bs, H, D]
    indexed by a per-request block_table. Our op consumes CONTIGUOUS per-kv-head
    [HKV, T, D]. The bridge (gather the request's local blocks into contiguous, and
    keep the peer shard contiguous on cuda:1) is provided for the single-sequence
    decode case in `_gather_contiguous`. Multi-sequence paged batching is left as a
    marked TODO and the impl FALLS BACK to the base kernel rather than guessing.
  * Peer KV is handed in by the allocator patch via `set_peer_kv` (same handoff as the
    copy-back path), as a contiguous [HKV, Tp, D] tensor on the peer device.

So: this backend is correct-or-fallback. It never silently returns wrong numbers; it
uses the fused op only when it can build valid contiguous inputs, else defers to the
parent FlashAttention impl.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

from .fused_attn import peer_fused_attn
from .paged import paged_cfk_decode

try:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    _HAVE_VLLM = True
except Exception:  # noqa: BLE001
    FlashAttentionImpl = object  # type: ignore
    _HAVE_VLLM = False


class PeerKVFusedAttentionImpl(FlashAttentionImpl):
    """FlashAttention impl that uses the multi-device CFK op for the decode phase
    (any batch size) when a peer paged KV cache is available; otherwise falls back.

    The decode path gathers each sequence's local/peer blocks into contiguous shards
    (umallm.peerkv.paged) and runs the exact multi-device merge. Prefill and any
    unexpected metadata defer to the base FlashAttention kernel (correct-or-fallback)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._peer_cache: Optional[torch.Tensor] = None   # [2, nb_p, bs, HKV, D] cuda:1
        self._peer_base: int = 1 << 30                     # global id >= this -> peer
        self._splits = int(os.environ.get("PEERKV_SPLITS", 16))
        self._scale = None

    def set_peer_kv(self, peer_cache: torch.Tensor, peer_base: int) -> None:
        """Allocator hands this layer its peer paged KV cache (cuda:1) and the global
        block-id threshold at/above which blocks live on the peer."""
        self._peer_cache = peer_cache
        self._peer_base = int(peer_base)

    # --- the CFK decode fast path ------------------------------------------

    def _try_fused(self, query, kv_cache, attn_metadata):
        """Return O shaped like `query` from the CFK op, or None to signal fallback."""
        if not _HAVE_VLLM or self._peer_cache is None:
            return None
        try:
            bt = getattr(attn_metadata, "block_table", None)      # [num_seqs, max_blk]
            seq_lens = getattr(attn_metadata, "seq_lens", None)   # [num_seqs]
            if bt is None or seq_lens is None:
                return None
            # decode only: exactly one query token per sequence.
            q = query
            if q.dim() == 3:                  # [num_tokens, H, D]
                num_tokens, H, D = q.shape
            elif q.dim() == 2:                # [H, D] single token
                num_tokens, (H, D) = 1, q.shape
                q = q.unsqueeze(0)
            else:
                return None
            num_seqs = bt.size(0) if hasattr(bt, "dim") and bt.dim() == 2 else 1
            if D != 128 or num_tokens != num_seqs:
                return None                   # prefill / chunked: defer
            if self._scale is None:
                self._scale = 1.0 / math.sqrt(D)
            bt2 = bt if bt.dim() == 2 else bt.unsqueeze(0)
            O = paged_cfk_decode(q, kv_cache, self._peer_cache, bt2, seq_lens,
                                 self._peer_base, scale=self._scale, splits=self._splits)
            return O                          # [num_seqs, H, D]
        except Exception:
            return None  # any surprise -> safe fallback

    # --- vLLM hook ----------------------------------------------------------

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):  # noqa: D401
        O = self._try_fused(query, kv_cache, attn_metadata)
        if O is None:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output)
        if query.dim() == 2:  # caller passed a single [H, D]
            O = O[0]
        if output is not None:
            output.copy_(O.to(output.dtype))
            return output
        return O.to(query.dtype)


def register_peerkv_fused() -> bool:
    """Monkeypatch FlashAttentionBackend.get_impl_cls -> PeerKVFusedAttentionImpl.
    Returns True if patched. Pair with the allocator patch that calls set_peer_kv with
    contiguous peer shards. (Dev/debug seam; the productized path is a formal
    KVConnectorBase_V1 backend, tracked in 03_track_C_product_runtime.md region 2.)"""
    try:
        from vllm.v1.attention.backends import flash_attn as v1fa
    except Exception:
        return False
    backend = getattr(v1fa, "FlashAttentionBackend", None)
    if backend is not None and hasattr(backend, "get_impl_cls"):
        @staticmethod
        def _impl_cls():
            return PeerKVFusedAttentionImpl
        backend.get_impl_cls = _impl_cls
        return True
    return False

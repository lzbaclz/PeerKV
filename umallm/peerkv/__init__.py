"""PeerKV product mainline (Track C): NVLink peer-GPU KV for vLLM.

This package hosts the Compute-Follows-KV (CFK) fast path:
  - ``fused_attn``: a multi-device CUDA op (split-K flash decode on a local + a peer
    KV shard, exact online-softmax merge), JIT-built from ``csrc/peer_fused_attn_ext.cu``.
  - ``vllm_backend``: a vLLM custom attention impl that calls it (Track C P3 entry).

Scope: A100/H100 NVLink, 2 GPUs, GQA, head_dim 128, batch=1 decode. See
``collaboration_plan/03_track_C_product_runtime.md`` (region 4 / P3).
"""
from __future__ import annotations

__all__ = ["load_fused_attn", "peer_fused_attn", "reference_attn"]

from .fused_attn import load_fused_attn, peer_fused_attn, reference_attn

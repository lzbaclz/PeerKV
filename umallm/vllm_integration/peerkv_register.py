"""Register the PeerKV attention backend without forking vLLM site-packages.

vLLM 0.8.5 selects the FlashAttention backend via the FLASH_ATTN enum; the backend
class exposes ``get_impl_cls() -> FlashAttentionImpl``. We monkeypatch that to return
our ``PeerKVFlashAttentionImpl`` (copy-back staging), so any run with
``VLLM_ATTENTION_BACKEND=FLASH_ATTN`` transparently gets the peer-GPU tier.

Usage (before constructing ``LLM`` / launching ``vllm serve``):
    import os; os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    from umallm.vllm_integration.peerkv_register import register_peerkv
    register_peerkv()

Pair with ``peer_kv_alloc.patch_peer_kv_allocation()`` (M2) which actually places a
block fraction on the peer GPU and hands each layer its peer tensor.

STATUS: import-safe everywhere; the monkeypatch target is verified to exist in
vLLM 0.8.5 (v1/attention/backends/flash_attn.py). Engine-level activation is
finalized on the dedicated box.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DONE = False


def register_peerkv() -> bool:
    """Swap FlashAttention's impl class for PeerKVFlashAttentionImpl. Idempotent;
    returns True if patched, False if vLLM unavailable."""
    global _DONE
    if _DONE:
        return True
    try:
        from vllm.v1.attention.backends import flash_attn as v1fa
        from .peerkv_attn import PeerKVFlashAttentionImpl
    except Exception as e:  # noqa: BLE001
        logger.info("PeerKV: vLLM v1 FlashAttention backend unavailable (%s); "
                    "register_peerkv() is a no-op.", e)
        return False

    backend = getattr(v1fa, "FlashAttentionBackend", None)
    if backend is not None and hasattr(backend, "get_impl_cls"):
        orig = backend.get_impl_cls

        @staticmethod
        def _peerkv_impl_cls():
            return PeerKVFlashAttentionImpl

        _peerkv_impl_cls._peerkv_orig = orig  # type: ignore[attr-defined]
        backend.get_impl_cls = _peerkv_impl_cls
        logger.info("PeerKV: FlashAttentionBackend.get_impl_cls -> "
                    "PeerKVFlashAttentionImpl")
        _DONE = True
        return True

    # fallback: rebind the module-level class so constructions pick it up
    if hasattr(v1fa, "FlashAttentionImpl"):
        v1fa.FlashAttentionImpl = PeerKVFlashAttentionImpl  # type: ignore[attr-defined]
        logger.info("PeerKV: rebound v1 FlashAttentionImpl (fallback path)")
        _DONE = True
        return True
    logger.warning("PeerKV: could not find a FlashAttention impl seam to patch.")
    return False

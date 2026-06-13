"""Registration seam for the CFK attention backend -- consolidated, not scattered.

Honest status on "de-monkeypatch":

  * What a vLLM **KV connector** (``KVConnectorBase_V1``, the ``--kv-transfer-config``
    mechanism) abstracts is KV *transfer/loading* (prefill->decode handoff, external
    KV reuse). CFK does NOT transfer KV; it replaces the attention *computation*
    (compute a partial on each GPU, merge). So CFK is an **attention backend**, not a
    connector -- the two are different vLLM seams. (The copy-back path in
    ``umallm/vllm_integration`` is the one that legitimately maps to a connector.)

  * vLLM selects an attention impl via ``FlashAttentionBackend.get_impl_cls()``. There
    are two ways to inject ours without editing site-packages:
      (A) override ``get_impl_cls`` once (what ``install_cfk`` does) -- a single,
          contained seam (vs. the old scattered monkeypatching), good for dev/bench;
      (B) ship an **out-of-tree platform/backend plugin** via a package entry point
          (``vllm.platform_plugins``), the productized route -- scaffolded in
          ``peerkv_platform.py`` + ``pyproject`` (entry point), to be finished and
          serve-tested on the dedicated 2-GPU box.

  * Making CFK actually run under ``vllm serve`` further needs the block manager to
    route a fraction of KV-block *writes* to the peer GPU so that
    ``PeerKVFusedAttentionImpl.set_peer_kv(peer_cache, peer_base)`` has real peer
    blocks to read. That block-manager integration + a real serve run are NOT done
    here and are tracked as M3 (region 2/3 in 03_track_C_product_runtime.md). The
    attention math, the multi-device op, and the paged gather are verified standalone
    (experiments/test_fused_attn_op.py, test_paged_cfk.py).

This module therefore gives a clean, single-seam dev registration + the plugin
scaffold, and is import-safe without vLLM.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_DONE = False


def install_cfk() -> bool:
    """Route vLLM's FlashAttention impl to PeerKVFusedAttentionImpl via the single
    ``get_impl_cls`` seam. Returns True if installed, False if vLLM is unavailable.

    This is the (A) dev/bench route. The original ``peerkv_register.py`` spread this
    across copy-back code; here it is one function with one override and one revert.
    """
    global _DONE
    if _DONE:
        return True
    try:
        from vllm.v1.attention.backends import flash_attn as v1fa
        from .vllm_backend import PeerKVFusedAttentionImpl
    except Exception as e:  # noqa: BLE001
        logger.info("PeerKV-CFK: vLLM unavailable (%s); install_cfk() is a no-op.", e)
        return False

    backend = getattr(v1fa, "FlashAttentionBackend", None)
    if backend is None or not hasattr(backend, "get_impl_cls"):
        logger.warning("PeerKV-CFK: no FlashAttentionBackend.get_impl_cls seam found.")
        return False

    orig = backend.get_impl_cls

    @staticmethod
    def _impl_cls():
        return PeerKVFusedAttentionImpl

    _impl_cls._peerkv_orig = orig  # type: ignore[attr-defined]
    backend.get_impl_cls = _impl_cls
    _DONE = True
    logger.info("PeerKV-CFK: FlashAttentionBackend.get_impl_cls -> PeerKVFusedAttentionImpl")
    return True


def uninstall_cfk() -> bool:
    """Revert install_cfk (restores vLLM's original get_impl_cls)."""
    global _DONE
    try:
        from vllm.v1.attention.backends import flash_attn as v1fa
        backend = getattr(v1fa, "FlashAttentionBackend", None)
        cur = getattr(backend, "get_impl_cls", None)
        orig = getattr(cur, "_peerkv_orig", None)
        if backend is not None and orig is not None:
            backend.get_impl_cls = orig
            _DONE = False
            return True
    except Exception:  # noqa: BLE001
        pass
    return False

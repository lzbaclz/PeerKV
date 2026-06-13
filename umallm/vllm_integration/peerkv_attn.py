"""PeerKV vLLM attention backend -- the copy-back staging wired into vLLM 0.8.5 v1.

This is the H100-side glue (the staging *logic* it calls is engine-independent and
unit-tested PASS in experiments/serve/m3_staging_test.py). It subclasses the v1
FlashAttention backend and inserts PeerKVStager.stage() in the exact window the
integration map identified:

  vllm/v1/attention/backends/flash_attn.py
    FlashAttentionImpl.forward            (:481)
      reshape_and_cache_flash(...)        (:527)   # writes new token's KV
      <-- PeerKV staging goes HERE (:536..:572) -->
      flash_attn_varlen_func(..., block_table=...) (:572)  # reads paged KV

We do not modify site-packages: select this backend via VLLM_ATTENTION_BACKEND or
global_force_attn_backend() (see peerkv_register.py).

STATUS: staging logic op-tested PASS; the vLLM hooks are CONFIRMED against the
installed 0.8.5 source -- forward(self, layer, query, key, value, kv_cache,
attn_metadata, output) with kv_cache=[2,num_blocks,block_size,H,D]
(flash_attn.py:481), attn_metadata.block_table (FlashAttentionMetadata:83),
Attention.impl set at layer.py:134. The only thing left for the dedicated box is
RUNTIME behaviour (does CUDA-graph replay reuse the metadata object / bake the
block_table?), flagged `# H100:` below.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from ..elastic_policy import (DoNoHarmViolation, Geometry, LinkState,
                              OperatingPoint, PeerState, enforce_do_no_harm)
from ..observability import metrics as _m
from .peerkv_staging import PeerKVLayout, PeerKVStager

logger = logging.getLogger("peerkv.attn")

try:  # import lazily so this module loads in a CPU sandbox / without vLLM
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl, FlashAttentionMetadata)
    _HAVE_VLLM = True
except Exception:  # noqa: BLE001
    FlashAttentionImpl = object       # type: ignore
    FlashAttentionMetadata = object   # type: ignore
    _HAVE_VLLM = False


def _layout_from_env(num_kv_heads: int, head_size: int, block_size: int) -> PeerKVLayout:
    """Layout from PEERKV_* env so we don't thread config through vLLM internals.
    PEERKV_C_LOCAL / PEERKV_C_PEER / PEERKV_C_SCRATCH are block counts; they are
    set by the allocator patch (peer_kv_alloc.py) once vLLM has profiled the ceiling."""
    g = lambda k, d: int(os.environ.get(k, d))
    return PeerKVLayout(
        c_local=g("PEERKV_C_LOCAL", 0), c_scratch=g("PEERKV_C_SCRATCH", 0),
        c_peer=g("PEERKV_C_PEER", 0), block_size=block_size,
        num_kv_heads=num_kv_heads, head_size=head_size,
        local_device=os.environ.get("PEERKV_LOCAL_DEV", "cuda:0"),
        peer_device=os.environ.get("PEERKV_PEER_DEV", "cuda:1"))


class PeerKVFlashAttentionImpl(FlashAttentionImpl):
    """FlashAttention with a peer-GPU KV tier: cold blocks live on cuda:1 and are
    coalesce-staged back into local scratch right before the paged kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # num_kv_heads / head_size are set by the base impl
        self._layout = _layout_from_env(self.num_kv_heads, self.head_size,
                                        block_size=int(os.environ.get("PEERKV_BLOCK", 16)))
        self._stager = PeerKVStager(self._layout)
        self._peer_kv: Optional[torch.Tensor] = None   # set by the allocator patch

    def _ensure_layout(self) -> None:
        """Attention impls are constructed at model build, BEFORE the allocator
        patch (peer_kv_alloc) publishes PEERKV_C_LOCAL/C_SCRATCH/C_PEER inside
        the wrapped initialize_kv_cache -- so __init__ freezes an all-zero
        layout (peer_base=0 would mark EVERY block peer-resident). Re-read the
        env lazily until a configured layout appears."""
        if self._layout.c_local == 0 and self._layout.c_peer == 0:
            fresh = _layout_from_env(self.num_kv_heads, self.head_size,
                                     block_size=self._layout.block_size)
            if fresh.c_local or fresh.c_peer:
                self._layout = fresh
                self._stager = PeerKVStager(fresh)

    def set_peer_kv(self, peer_kv: torch.Tensor) -> None:
        """Called by peer_kv_alloc.py with this layer's cuda:1 KV tensor."""
        self._peer_kv = peer_kv

    def _guard_route(self) -> None:
        """Hot-path do-no-harm tripwire (04_cross_cutting SS1 Step 2).

        Staging from the peer tier is the COPYBACK corner; the enforceable
        rule at this depth is R1-route (link health -- R1-fit/R1-busy are
        admission-time and already gated by the selector). The link state is
        read from PEERKV_NVLINK_GBPS (set by the runtime's probe; no
        subprocess on the hot path). PEERKV_STRICT=1 raises (CI); production
        warns once + increments peerkv_do_no_harm_violations_total and
        proceeds -- blocks already on the peer cannot be legally dropped
        mid-flight, the admission layer owns the repair."""
        nvl = float(os.environ.get("PEERKV_NVLINK_GBPS", LinkState.nvlink_eff_gbps))
        link = LinkState(nvlink_eff_gbps=nvl)
        try:
            enforce_do_no_harm(OperatingPoint.COPYBACK, ctx_tokens=1 << 62,
                               geom=Geometry.llama2_7b_mha(),
                               peer=PeerState(nvlink_bw_gbps=nvl), link=link)
        except DoNoHarmViolation as e:
            _m.DO_NO_HARM_VIOL.labels(rule="R1-route").inc()
            if os.environ.get("PEERKV_STRICT", "0") == "1":
                raise
            if not getattr(self, "_route_warned", False):
                self._route_warned = True
                logger.warning("hot-path do-no-harm: %s (staging proceeds; "
                               "admission layer must reroute new requests)", e)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):  # noqa: D401
        # kv_cache here is the local(+scratch) tensor on cuda:0. After the base
        # impl writes the new token (reshape_and_cache_flash), stage peer blocks and
        # remap the block_table, then let the unmodified kernel run over cuda:0.
        if self._peer_kv is None or not _HAVE_VLLM:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output)
        self._ensure_layout()
        if self._layout.c_local == 0 and self._layout.c_peer == 0:
            # allocator has not published a layout yet: treat as all-local
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output)

        bt = getattr(attn_metadata, "block_table", None)   # H100: confirm field name
        if bt is not None and bool((bt >= self._layout.peer_base).any()):
            self._guard_route()
            remapped = self._stager.stage(kv_cache, self._peer_kv, bt)
            # shallow-copy metadata with the remapped block_table so we don't mutate
            # vLLM's cached object across CUDA-graph replays
            attn_metadata = _with_block_table(attn_metadata, remapped)  # H100: wire
        return super().forward(layer, query, key, value, kv_cache, attn_metadata, output)


def _with_block_table(meta, block_table):
    """Return a copy of the attn metadata with block_table replaced.
    H100: vLLM's FlashAttentionMetadata is a dataclass; use dataclasses.replace,
    or set the attribute on a shallow copy if it is frozen."""
    import copy
    m = copy.copy(meta)
    setattr(m, "block_table", block_table)
    return m

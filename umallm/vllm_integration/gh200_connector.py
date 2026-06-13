"""UMA-LLM Grace-Hopper connector for vLLM V1.

Implements vLLM's ``KVConnectorBase_V1`` (same interface the author's
OrchKvCache connector targets) to demote cold KV blocks to the Grace
LPDDR5X tier (coherent over NVLink-C2C) and/or KIVI-compress them in place,
driven by :class:`umallm.vllm_integration.placement.GraceHopperPlacement`.

Two roles:
  * SCHEDULER: per step, scores each request's blocks and uses the GH200
    cost model + sizing inversion to decide which blocks move to ``grace``
    / ``compressed`` and which to restore to ``hbm``.
  * WORKER: performs the moves. ``grace`` demotion is, by default, a
    zero-copy residency hint (Route B: cudaMemAdvise + cudaMemPrefetchAsync
    via :class:`umallm.uma_alloc.UMAResidencyController`) when a managed KV
    pool is live; otherwise it falls back to a side-stream HBM->host copy
    (Route A, gated by ``coherent_read``). ``compressed`` demotion
    KIVI-quantizes the block; restores reverse these.

Route B (the allocator-layer integration that makes a "tier" a page residency
hint rather than a copy) is wired in :mod:`umallm.vllm_integration.uma_backend`
-- call ``patch_vllm_kv_allocation()`` before the engine builds its KV cache.

Launch (on a GH200 box, after `pip install vllm`):
    vllm serve <model> \
      --kv-transfer-config '{"kv_connector":"UMAGraceHopperConnector",
        "kv_connector_module_path":"umallm.vllm_integration.gh200_connector",
        "kv_role":"kv_both",
        "kv_connector_extra_config":{"deadline_ms":50,"miss_target":0.01,
          "grace_pool_gb":128,"cold_bits":4}}'

UNTESTED off-hardware: requires CUDA + vLLM + a GH200 (or any coherent
CPU-GPU box). The scheduler placement logic is unit-tested on CPU
(tests/test_vllm_placement.py); the worker device path is the remaining
on-hardware validation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from .placement import GH200PlacementConfig, GraceHopperPlacement
from ..compression import dequantize_block, quantize_block, quantize_block_2bit
from ..pressure import PressureLevel
from ..uma_alloc import UMAResidencyController, native_available

logger = logging.getLogger(__name__)

# --- guarded torch (absent in a CPU sandbox) ---------------------------- #
try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False
    torch = None  # type: ignore

# --- guarded vLLM V1 connector base (same pattern as OrchKvCache) ------- #
try:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
    )
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

    class KVConnectorRole:  # type: ignore
        SCHEDULER = "scheduler"
        WORKER = "worker"

    class KVConnectorMetadata:  # type: ignore
        pass

    class KVConnectorBase_V1:  # type: ignore
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._connector_metadata = None
            self._role = role

        @property
        def role(self):
            return self._role

        def bind_connector_metadata(self, meta):
            self._connector_metadata = meta

        def clear_connector_metadata(self):
            self._connector_metadata = None

if TYPE_CHECKING:
    from vllm.config import VllmConfig, KVCacheConfig
    from vllm.v1.core.kv_cache_utils import KVCacheBlocks
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext


@dataclass
class UMAGraceHopperMetadata(KVConnectorMetadata):
    """Scheduler -> worker per-step plan. {layer_name: [block_id, ...]}."""

    blocks_to_grace: dict[str, list[int]] = field(default_factory=dict)
    blocks_to_compress: dict[str, list[int]] = field(default_factory=dict)
    blocks_to_restore: dict[str, list[int]] = field(default_factory=dict)
    # old_block_id -> new_block_id for a preempted request that resumed into
    # different slots. The worker re-keys its held buffers (and residency
    # state) by this map *before* the restore, so a held KV lands in the new
    # slot. Empty in the common (no-preemption) case.
    remap: dict[int, int] = field(default_factory=dict)


# ====================================================================== #
# Worker
# ====================================================================== #
class _Worker:
    """Performs HBM<->Grace coherent moves and KIVI (de)compression."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.cold_bits = int(cfg.get("cold_bits", 4))
        self.coherent_read = bool(cfg.get("coherent_read", True))
        self._kv: dict[str, "torch.Tensor"] = {}
        self._grace: dict[tuple[str, int], "torch.Tensor"] = {}
        self._compressed: dict[tuple[str, int], tuple] = {}
        self._move = None
        if HAS_TORCH:
            self._move = torch.cuda.Stream() if torch.cuda.is_available() else None
        # Route B: when a managed KV pool is live, T0<->T1 is a zero-copy
        # residency hint (cudaMemAdvise + prefetch) rather than a host copy.
        # Defaults on when the native extension + device support it; force
        # with cfg["residency_hints"]. Route A (copy) is the fallback.
        self._residency_enabled = bool(cfg.get("residency_hints",
                                               native_available()))
        self._block_dim = int(cfg.get("kv_block_dim", 1))
        self._residency: UMAResidencyController | None = None
        self.stats = {"to_grace": 0, "to_compressed": 0, "restored": 0}

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        self._kv = kv_caches
        if self._residency_enabled:
            self._residency = UMAResidencyController(block_dim=self._block_dim)
            self._residency.register_kv_caches(kv_caches)
        logger.info("UMA-GH200: registered %d KV layers (residency_hints=%s)",
                    len(kv_caches), self._residency_enabled)

    def _stream_ptr(self) -> int:
        s = getattr(self, "_move", None)
        if s is not None and hasattr(s, "cuda_stream"):
            return int(s.cuda_stream)
        return 0

    # -- demotions ----------------------------------------------------- #
    def to_grace(self, layer: str, block_ids: list[int]):
        """Demote to the Grace LPDDR5X tier.

        Route B (managed pool): a residency hint -- prefer + prefetch the
        block's pages onto Grace. No bytes are staged; the virtual address
        and block table are untouched, and a coherent C2C read serves the
        page in place. Route A fallback: a side-stream HBM->host copy.
        """
        if not block_ids:
            return
        if self._residency is not None:
            self._residency.to_grace(layer, block_ids, stream=self._stream_ptr())
            self.stats["to_grace"] += len(block_ids)
            return
        kv = self._kv.get(layer)
        if kv is None:
            return
        stream = getattr(self, "_move", None)
        ctx = torch.cuda.stream(stream) if stream is not None else _nullctx()
        with ctx:
            for bid in block_ids:
                key = (layer, bid)
                if key not in self._grace:
                    # Grace-resident (host) buffer; coherent on GH200.
                    self._grace[key] = torch.empty_like(kv[bid], device="cpu")
                self._grace[key].copy_(kv[bid], non_blocking=True)
        self.stats["to_grace"] += len(block_ids)

    def to_compressed(self, layer: str, block_ids: list[int]):
        """HBM -> KIVI 4-bit (in place; frees the fp16 footprint)."""
        kv = self._kv.get(layer)
        if kv is None or not block_ids:
            return
        for bid in block_ids:
            arr = _to_numpy(kv[bid]).reshape(-1, kv[bid].shape[-1])
            q = (quantize_block_2bit(arr) if self.cold_bits == 2
                 else quantize_block(arr))
            self._compressed[(layer, bid)] = (q, tuple(kv[bid].shape))
        self.stats["to_compressed"] += len(block_ids)

    # -- preemption remap ---------------------------------------------- #
    def apply_remap(self, remap: dict[int, int]):
        """Re-key worker-held KV from old slot ids to the resumed request's new
        slot ids, so the following ``restore`` lands in the right slot.

        Sound for the *buffer-backed* tiers -- ``_grace`` (Route A host copy)
        and ``_compressed`` (KIVI bytes) genuinely hold the KV, so moving the
        key and restoring recovers a preempted request without recompute. For
        Route B residency hints there is no separate buffer: the managed pages
        of a freed slot are gone, so we only re-key the state machine; those
        blocks fall back to recompute (vLLM's default). Route B's intended
        behavior under pressure is to *demote* (keep the request resident on
        Grace, same ids) rather than let vLLM preempt -- then no remap is
        needed.
        """
        if not remap:
            return

        def _rekey(d):
            moved = {}
            for (layer, bid), v in list(d.items()):
                if bid in remap:
                    del d[(layer, bid)]
                    moved[(layer, remap[bid])] = v
            d.update(moved)

        _rekey(self._grace)
        _rekey(self._compressed)
        if self._residency is not None and hasattr(self._residency, "remap"):
            self._residency.remap(remap)
        self.stats.setdefault("remapped", 0)
        self.stats["remapped"] += len(remap)

    # -- restores ------------------------------------------------------ #
    def restore(self, layer: str, block_ids: list[int]):
        if not block_ids:
            return
        kv = self._kv.get(layer)
        for bid in block_ids:
            key = (layer, bid)
            if key in self._compressed:               # T2 -> HBM (dequant)
                if kv is None:
                    continue
                q, shape = self._compressed.pop(key)
                arr = dequantize_block(q).reshape(shape)
                kv[bid].copy_(_from_numpy(arr, kv[bid]))
            elif self._residency is not None and \
                    self._residency.residency_of(layer, bid) == "grace":
                # Route B: prefetch the pages back to HBM (no copy-back).
                self._residency.to_hbm(layer, [bid], stream=self._stream_ptr())
            elif key in self._grace:                  # Route A copy-back
                if not self.coherent_read and kv is not None:
                    kv[bid].copy_(self._grace[key], non_blocking=True)
                # coherent_read -> read-through, no copy needed on GH200
                self._grace.pop(key, None)
        self.stats["restored"] += len(block_ids)

    def wait(self):
        if HAS_TORCH and getattr(self, "_move", None) is not None:
            self._move.synchronize()

    def footprint(self) -> dict:
        comp_bytes = sum(q.nbytes() for q, _ in self._compressed.values())
        fp = {"grace_blocks": len(self._grace),
              "compressed_blocks": len(self._compressed),
              "compressed_bytes": comp_bytes, **self.stats}
        if self._residency is not None:
            # Route B: grace residency lives in the controller, not _grace.
            fp["residency"] = self._residency.footprint()
        return fp


# ====================================================================== #
# Scheduler
# ====================================================================== #
class _Scheduler:
    """Decides per-step demotions/restores from the GH200 cost model."""

    def __init__(self, cfg: dict[str, Any], layer_names: list[str] | None = None):
        pcfg = GH200PlacementConfig(
            deadline_ms=float(cfg.get("deadline_ms", 50.0)),
            miss_target=float(cfg.get("miss_target", 1e-2)),
            block_size_tokens=int(cfg.get("tokens_per_block", 16)),
        )
        self.placement = GraceHopperPlacement(cfg=pcfg)
        self.n_layers = int(cfg.get("n_layers", 32))
        # Layer names must match the worker's registered KV-cache keys (the
        # plan is applied per layer). On real hardware these come from the
        # KVCacheConfig; off-hardware we synthesize stand-ins for tests.
        self._layer_names: list[str] = list(layer_names) if layer_names else \
            [f"layer_{i}" for i in range(self.n_layers)]
        # per-request EMA attention mass per block (hotness)
        self._hotness: dict[str, np.ndarray] = {}
        self._req_blocks: dict[str, list[int]] = {}
        # Last tier name committed per (req_id -> {block_id: tier}). The plan
        # is a *delta* against this, so a block already resident at a tier is
        # not re-moved every decode step, and a reheated block is restored.
        self._prev_tier: dict[str, dict[int, str]] = {}
        self._plan = UMAGraceHopperMetadata()
        self._block_size = int(cfg.get("tokens_per_block", 16))
        self._enable_reuse = bool(cfg.get("enable_external_reuse", True))
        # Blocks vLLM re-allocated for a resumed request whose KV we still
        # hold demoted (grace/compressed): force a restore on the next plan
        # step, overriding whatever the steady-state cost model wants.
        self._force_restore: dict[str, set[int]] = {}

    def observe_attention(self, req_id: str, block_mass: np.ndarray, beta: float = 0.7):
        prev = self._hotness.get(req_id)
        m = np.asarray(block_mass, dtype=np.float32)
        self._hotness[req_id] = m if prev is None or prev.shape != m.shape \
            else beta * prev + (1 - beta) * m

    def update_block_table(self, req_id: str, block_ids: list[int], *, append: bool):
        """Maintain the full block table from vLLM's per-step deltas.

        ``SchedulerOutput`` only carries *new* blocks for a decoding request,
        so we accumulate them; new requests reset the table.
        """
        if append:
            self._req_blocks.setdefault(req_id, []).extend(int(b) for b in block_ids)
        else:
            self._req_blocks[req_id] = [int(b) for b in block_ids]

    def forget_request(self, req_id: str):
        self._hotness.pop(req_id, None)
        self._req_blocks.pop(req_id, None)
        self._prev_tier.pop(req_id, None)
        self._force_restore.pop(req_id, None)

    # -- external-KV reuse (vLLM get_num_new_matched_tokens / alloc) ---- #
    def num_new_matched_tokens(self, req_id: str, num_computed_tokens: int):
        """Tokens this connector can supply for a (re)scheduled request.

        On UMA a demoted block is *not* evicted: it stays resident and
        coherent on the SoC (grace/compressed), so a request we already track
        can be resumed without recomputing its KV. We report the held tokens
        beyond what vLLM has locally; vLLM allocates slots for them and the
        worker materialises them on ``start_load_kv`` (hence async=True).

        A first-seen request is untracked (``_req_blocks`` empty) -> (0, False),
        so this never fabricates a prefix for genuinely new work. Assumes the
        request keeps its block ids across the demote/restore window; a
        cross-preemption resume needs an old->new block-id remap on the worker
        side (the remaining hardware-path TODO).
        """
        if not self._enable_reuse:
            return 0, False
        held = len(self._req_blocks.get(req_id, [])) * self._block_size
        new = held - int(num_computed_tokens)
        if new <= 0:
            return 0, False
        return new, True

    def note_external_match(self, req_id: str, block_ids: list[int]):
        """Record a resumed request's (re)allocation and queue its demoted
        blocks for a forced restore on the next plan step.

        If the request was preempted and resumed into *different* slots, build
        a positional old->new remap (the request's logical block k held slot
        ``old[k]`` and now holds ``new[k]``), stash it in the plan so the
        worker re-keys its held buffers, and carry the tier state onto the new
        ids. If the ids are unchanged (demote-while-resident, the common Route
        B case) the remap is empty and this is just a forced restore.
        """
        new = [int(b) for b in block_ids]
        old = self._req_blocks.get(req_id)
        prev = self._prev_tier.get(req_id, {})
        if old and new:
            n = min(len(old), len(new))
            remap = {int(old[k]): new[k] for k in range(n)
                     if int(old[k]) != new[k]}
            if remap:
                self._plan.remap.update(remap)
                # carry tier state from old slots onto the new ones
                prev = {new[k]: prev.get(int(old[k]), "hbm") for k in range(n)}
                self._prev_tier[req_id] = prev
        if new:
            self.update_block_table(req_id, new, append=False)
        demoted = {b for b, t in prev.items() if t != "hbm"}
        if demoted:
            self._force_restore.setdefault(req_id, set()).update(demoted)

    def plan_request(self, req_id: str, block_ids: list[int], layer_names: list[str],
                     pressure: PressureLevel = PressureLevel.NORMAL):
        self._req_blocks[req_id] = list(block_ids)
        hot = self._hotness.get(req_id)
        if hot is None or hot.shape[0] != len(block_ids):
            hot = np.arange(len(block_ids), dtype=np.float32)  # recency proxy
        assign, sizing = self.placement.assign(block_ids, hot, pressure)
        # A resumed request's demoted blocks must come back this step
        # regardless of what the steady-state cost model wants; pin them to
        # "hbm" so the delta below emits a restore (and never a re-demote).
        forced = self._force_restore.pop(req_id, None)
        if forced:
            for bid in forced:
                if bid in assign:
                    assign[bid] = "hbm"
        prev = self._prev_tier.get(req_id, {})
        # Emit only tier *transitions*. A block defaults to "hbm" until moved,
        # so first-seen demotions still fire; an unchanged tier emits nothing;
        # a block returning to "hbm" (reheated, or pulled into sink/window) is
        # queued for restore.
        for ln in layer_names:
            g = self._plan.blocks_to_grace.setdefault(ln, [])
            c = self._plan.blocks_to_compress.setdefault(ln, [])
            r = self._plan.blocks_to_restore.setdefault(ln, [])
            for bid, tier in assign.items():
                if tier == prev.get(bid, "hbm"):
                    continue
                if tier == "grace":
                    g.append(bid)
                elif tier == "compressed":
                    c.append(bid)
                elif tier == "hbm":
                    r.append(bid)
        self._prev_tier[req_id] = {int(b): t for b, t in assign.items()}
        return sizing

    def plan_step(self, active_req_ids, pressure: PressureLevel = PressureLevel.NORMAL):
        """Plan every request scheduled this step, then hand off the delta."""
        for rid in active_req_ids:
            blocks = self._req_blocks.get(rid)
            if blocks:
                self.plan_request(rid, blocks, self._layer_names, pressure)
        return self.take_plan()

    def take_plan(self) -> UMAGraceHopperMetadata:
        plan, self._plan = self._plan, UMAGraceHopperMetadata()
        return plan


# ====================================================================== #
# Connector
# ====================================================================== #
class UMAGraceHopperConnector(KVConnectorBase_V1):
    """vLLM V1 connector: UMA-LLM residency tiering on Grace-Hopper."""

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True

    def __init__(self, vllm_config: "VllmConfig", role: "KVConnectorRole",
                 kv_cache_config: "KVCacheConfig | None" = None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = {}
        ktc = getattr(vllm_config, "kv_transfer_config", None)
        if ktc is not None and getattr(ktc, "kv_connector_extra_config", None):
            extra = ktc.kv_connector_extra_config
        mc = getattr(vllm_config, "model_config", None)
        cfg = {
            "deadline_ms": extra.get("deadline_ms", 50.0),
            "miss_target": extra.get("miss_target", 1e-2),
            "cold_bits": extra.get("cold_bits", 4),
            "coherent_read": extra.get("coherent_read", True),
            "tokens_per_block": extra.get("tokens_per_block", 16),
            "enable_external_reuse": extra.get("enable_external_reuse", True),
            "n_layers": getattr(mc, "num_layers", extra.get("n_layers", 32)),
        }
        self._worker = _Worker(cfg) if role == KVConnectorRole.WORKER else None
        layer_names = _layer_names_from_config(kv_cache_config, cfg["n_layers"])
        self._scheduler = (_Scheduler(cfg, layer_names)
                           if role == KVConnectorRole.SCHEDULER else None)
        logger.info("UMAGraceHopperConnector init (role=%s)", role)

    # ----- worker side ----------------------------------------------- #
    def register_kv_caches(self, kv_caches):
        assert self._worker is not None
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs):
        assert self._worker is not None
        meta = self._connector_metadata
        if isinstance(meta, UMAGraceHopperMetadata):
            if meta.remap:                       # preempted-resume: re-key first
                self._worker.apply_remap(meta.remap)
            for ln, bids in meta.blocks_to_restore.items():
                self._worker.restore(ln, bids)

    def wait_for_layer_load(self, layer_name: str):
        assert self._worker is not None
        self._worker.wait()

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        assert self._worker is not None
        meta = self._connector_metadata
        if isinstance(meta, UMAGraceHopperMetadata):
            self._worker.to_grace(layer_name, meta.blocks_to_grace.get(layer_name, []))
            self._worker.to_compressed(layer_name, meta.blocks_to_compress.get(layer_name, []))

    def wait_for_save(self):
        assert self._worker is not None
        self._worker.wait()

    def get_stats(self):
        return self._worker.footprint() if self._worker else {}

    # ----- scheduler side -------------------------------------------- #
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        assert self._scheduler is not None
        return self._scheduler.num_new_matched_tokens(
            _request_id(request), int(num_computed_tokens))

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if self._scheduler is None or int(num_external_tokens) <= 0:
            return
        self._scheduler.note_external_match(
            _request_id(request), _blocks_to_ids(blocks))

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        assert self._scheduler is not None
        # New requests carry their full block table; decoding requests carry
        # only the blocks appended this step. Accumulate both, then plan the
        # set actually scheduled this step.
        for req_id, block_ids in _iter_new_reqs(scheduler_output):
            self._scheduler.update_block_table(req_id, block_ids, append=False)
        for req_id, new_block_ids in _iter_cached_reqs(scheduler_output):
            self._scheduler.update_block_table(req_id, new_block_ids, append=True)
        return self._scheduler.plan_step(_active_req_ids(scheduler_output))

    def request_finished(self, request, block_ids):
        if self._scheduler is not None:
            rid = _request_id(request)
            if rid is not None:
                self._scheduler.forget_request(rid)
        return False, None


# ---------------------------------------------------------------------- #
# SchedulerOutput parsing. vLLM's V1 SchedulerOutput shape has drifted
# across releases (block_ids went nested per-KV-group; cached reqs went
# from a list of structs to one struct with parallel lists), so we read it
# defensively via getattr. Assumed fields: ``scheduled_new_reqs`` (each with
# ``req_id`` + ``block_ids``), ``scheduled_cached_reqs``, and
# ``num_scheduled_tokens`` (req_id -> int) for the active set.
def _layer_names_from_config(kv_cache_config, n_layers) -> list[str]:
    names: list[str] = []
    for g in getattr(kv_cache_config, "kv_cache_groups", None) or []:
        names.extend(getattr(g, "layer_names", None) or [])
    return names or [f"layer_{i}" for i in range(int(n_layers))]


def _normalize_block_ids(block_ids) -> list[int]:
    """Flatten vLLM's optionally per-KV-group nested block ids to a flat list."""
    if block_ids is None:
        return []
    seq = list(block_ids)
    if seq and isinstance(seq[0], (list, tuple)):
        return [int(b) for grp in seq for b in grp]
    return [int(b) for b in seq]


def _request_id(request):
    return getattr(request, "request_id", None) or getattr(request, "req_id", None)


def _blocks_to_ids(blocks) -> list[int]:
    """Extract flat block ids from vLLM's ``KVCacheBlocks`` (or a plain list).

    vLLM has exposed the allocated blocks both as a ``block_ids`` attribute and
    via ``get_block_ids()``/``get_unhashed_block_ids()`` across releases, so we
    probe defensively and fall back to treating ``blocks`` as a raw id list.
    """
    if blocks is None:
        return []
    for attr in ("get_block_ids", "get_unhashed_block_ids", "block_ids"):
        v = getattr(blocks, attr, None)
        if callable(v):
            try:
                v = v()
            except TypeError:
                continue
        if v is not None:
            return _normalize_block_ids(v)
    try:
        return _normalize_block_ids(blocks)
    except (TypeError, ValueError):
        return []


def _iter_new_reqs(so):
    for r in getattr(so, "scheduled_new_reqs", None) or []:
        rid = getattr(r, "req_id", None)
        if rid is not None:
            yield rid, _normalize_block_ids(getattr(r, "block_ids", None))


def _iter_cached_reqs(so):
    cached = getattr(so, "scheduled_cached_reqs", None)
    if not cached:
        return
    rids = getattr(cached, "req_ids", None)
    if rids is not None:  # newer V1: one struct with parallel lists
        new_blocks = getattr(cached, "new_block_ids", None) or [None] * len(rids)
        for rid, nb in zip(rids, new_blocks):
            yield rid, _normalize_block_ids(nb)
        return
    for r in cached:  # older V1: a list of per-request structs
        rid = getattr(r, "req_id", None)
        if rid is not None:
            yield rid, _normalize_block_ids(getattr(r, "new_block_ids", None))


def _active_req_ids(so) -> list[str]:
    nst = getattr(so, "num_scheduled_tokens", None)
    if isinstance(nst, dict) and nst:
        return list(nst.keys())
    ids = [rid for rid, _ in _iter_new_reqs(so)]
    ids += [rid for rid, _ in _iter_cached_reqs(so)]
    return ids


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _to_numpy(t):
    if HAS_TORCH and hasattr(t, "detach"):
        return t.detach().to("cpu", dtype=torch.float32).numpy()
    return np.asarray(t, dtype=np.float32)


def _from_numpy(arr, like):
    if HAS_TORCH and hasattr(like, "dtype"):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(like.device, dtype=like.dtype)
    return arr

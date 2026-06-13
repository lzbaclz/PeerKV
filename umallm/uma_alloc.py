"""Route B: real UMA residency control at the allocator layer (GH200).

This is the part that makes UMA-LLM *not* an offload connector. Instead of
copying cold KV blocks into a host staging buffer, we:

1. Allocate the KV pool as CUDA Unified Memory (``cudaMallocManaged``) by
   routing vLLM's KV allocation through a :class:`torch.cuda.MemPool` bound
   to our pluggable allocator (:class:`UMAManagedAllocator`). Only the KV
   pool is managed; weights/activations keep the fast default allocator.
2. Treat a "tier" as a *residency hint* on the block's pages:
   ``to_grace`` = ``cudaMemAdvise(PreferredLocation=Grace)`` +
   ``cudaMemPrefetchAsync(Grace)``; ``to_hbm`` = the same toward HBM.
   The block's virtual address never changes, so the attention block table
   and kernel are untouched -- T0<->T1 is genuinely zero-copy, and a
   not-yet-migrated read is served coherently over NVLink-C2C.

The block->page-range arithmetic, the residency state machine, and the
GB-on-Grace accounting are pure Python and unit-tested on CPU. The native
calls (:mod:`umallm._uma_native`) are thin and guarded: when the extension
or CUDA is absent we keep all the bookkeeping and simply skip the device
hint (a simulation), so the logic is exercisable anywhere.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --- guarded native extension ------------------------------------------- #
try:
    from . import _uma_native as _native  # type: ignore

    _HAS_NATIVE = True
except Exception:  # pragma: no cover - depends on the build host
    _native = None
    _HAS_NATIVE = False


def native_available() -> bool:
    """True when the CUDA extension is built *and* the device supports it."""
    return bool(_HAS_NATIVE and _native is not None and _native.available())


def regime() -> str:
    """Which memory regime the native layer is actually running on.

    - ``"coherent_uma"``: coherent CPU-GPU unified memory (GH200 NVLink-C2C) --
      a demotion is a genuine zero-copy residency hint, the paper's headline
      regime.
    - ``"discrete_migration"``: discrete CUDA (e.g. A100 over PCIe) -- the same
      residency hint triggers a real page migration. This is the *non-coherent
      / discrete reference*, NOT coherent UMA; report it as such.
    - ``"simulation"``: no native extension (CPU/Mac) -- bookkeeping only.
    """
    if not native_available():
        return "simulation"
    try:
        return "coherent_uma" if _native.coherent() else "discrete_migration"
    except Exception:  # pragma: no cover - older CUDA without the attribute
        return "discrete_migration"


def device_name() -> str:
    """CUDA device name, or a sentinel off-hardware."""
    if not native_available():
        return "cpu-sim"
    try:
        return str(_native.device_name())
    except Exception:  # pragma: no cover
        return "unknown-cuda-device"


# --- array adapters (torch tensor or numpy ndarray) --------------------- #
def _data_ptr(t) -> int:
    fn = getattr(t, "data_ptr", None)
    if callable(fn):
        return int(fn())                       # torch
    return int(t.ctypes.data)                   # numpy


def _itemsize(t) -> int:
    fn = getattr(t, "element_size", None)
    if callable(fn):
        return int(fn())                        # torch
    return int(t.itemsize)                       # numpy


def _strides_bytes(t) -> tuple[int, ...]:
    st = t.stride() if hasattr(t, "stride") else None
    if st is not None:                           # torch: strides in elements
        es = _itemsize(t)
        return tuple(int(s) * es for s in st)
    return tuple(int(s) for s in t.strides)      # numpy: strides in bytes


def _shape(t) -> tuple[int, ...]:
    return tuple(int(d) for d in t.shape)


def _is_contiguous(t) -> bool:
    fn = getattr(t, "is_contiguous", None)
    if callable(fn):
        return bool(fn())                        # torch
    flags = getattr(t, "flags", None)
    return bool(flags["C_CONTIGUOUS"]) if flags is not None else True


# ====================================================================== #
# Pluggable allocator (the allocator-layer hook)
# ====================================================================== #
class UMAManagedAllocator:
    """Binds torch to the ``cudaMallocManaged`` C ABI in ``_uma_native``.

    Use :meth:`mem_pool` to get a :class:`torch.cuda.MemPool` and allocate
    *only* the KV cache inside ``with torch.cuda.use_mem_pool(pool):`` -- see
    :mod:`umallm.vllm_integration.uma_backend`.
    """

    def __init__(self):
        if not _HAS_NATIVE or _native is None:
            raise RuntimeError(
                "umallm._uma_native is not built; rebuild with "
                "UMA_BUILD_CUDA=1 on a CUDA host."
            )
        import torch

        self._torch = torch
        so_path = _native.__file__
        self._alloc = torch.cuda.memory.CUDAPluggableAllocator(
            so_path, "uma_malloc", "uma_free"
        )

    def allocator(self):
        return self._alloc.allocator()

    def mem_pool(self):
        """A MemPool that allocates managed memory (scope KV here)."""
        return self._torch.cuda.MemPool(self._alloc.allocator())

    def install_global(self):
        """Route *all* CUDA allocations through managed memory.

        Discouraged on GH200 (weights/activations would fault-thrash); prefer
        the scoped :meth:`mem_pool`. Must run before any CUDA allocation.
        """
        self._torch.cuda.memory.change_current_allocator(self._alloc)


# ====================================================================== #
# Residency controller (the brain -- pure Python, CPU-testable)
# ====================================================================== #
@dataclass
class UMAResidencyController:
    """Maps KV blocks to managed page ranges and issues residency hints.

    ``block_dim`` is the axis of each layer's KV tensor indexed by vLLM block
    id (0 for ``(num_blocks, ...)``; 1 for the FlashAttention
    ``(2, num_blocks, block_size, heads, head_dim)`` layout).
    """

    block_dim: int = 0
    # (layer, block_id) -> "grace" | "hbm". Absent == "hbm" (born HBM-hot).
    _state: dict[tuple[str, int], str] = field(default_factory=dict)
    _kv: dict[str, Any] = field(default_factory=dict)
    _layer_block_nbytes: dict[str, int] = field(default_factory=dict)
    _n_blocks: dict[str, int] = field(default_factory=dict)
    stats: dict[str, int] = field(
        default_factory=lambda: {"to_grace": 0, "to_hbm": 0, "hint_errors": 0})

    # -- registration -------------------------------------------------- #
    def register_kv_caches(self, kv_caches: dict[str, Any]):
        self._kv = dict(kv_caches)
        for layer, t in kv_caches.items():
            shape = _shape(t)
            if not 0 <= self.block_dim < len(shape):
                raise ValueError(
                    f"block_dim={self.block_dim} out of range for layer "
                    f"{layer!r} with shape {shape}")
            n_blocks = shape[self.block_dim]
            total = math.prod(shape) * _itemsize(t)
            self._n_blocks[layer] = n_blocks
            self._layer_block_nbytes[layer] = total // n_blocks if n_blocks else 0
            if native_available() and not _is_contiguous(t):
                logger.warning(
                    "UMA residency: layer %s KV tensor is non-contiguous; "
                    "page-range hints assume C-contiguity.", layer)
        logger.info("UMA residency: registered %d KV layers", len(kv_caches))

    def block_nbytes(self, layer: str) -> int:
        return self._layer_block_nbytes.get(layer, 0)

    # -- page-range arithmetic ----------------------------------------- #
    def _block_ranges(self, layer: str, block_id: int) -> list[tuple[int, int]]:
        """Contiguous (ptr, nbytes) slabs backing one block.

        Dims outer to ``block_dim`` (e.g. the K/V axis) index separate slabs;
        dims inner to it are the per-block contiguous payload.
        """
        t = self._kv[layer]
        shape = _shape(t)
        strides = _strides_bytes(t)
        base = _data_ptr(t)
        itemsize = _itemsize(t)
        inner_elems = math.prod(shape[self.block_dim + 1:]) if \
            self.block_dim + 1 < len(shape) else 1
        inner_nbytes = inner_elems * itemsize
        block_off = block_id * strides[self.block_dim]

        outer_dims = shape[: self.block_dim]
        if not outer_dims:
            return [(base + block_off, inner_nbytes)]
        ranges: list[tuple[int, int]] = []
        # Cartesian product over outer-dim indices.
        idx = [0] * len(outer_dims)
        while True:
            off = block_off + sum(idx[k] * strides[k] for k in range(len(idx)))
            ranges.append((base + off, inner_nbytes))
            # increment mixed-radix counter
            d = len(outer_dims) - 1
            while d >= 0:
                idx[d] += 1
                if idx[d] < outer_dims[d]:
                    break
                idx[d] = 0
                d -= 1
            if d < 0:
                break
        return ranges

    # -- residency moves (the "tier change") --------------------------- #
    def _apply(self, layer: str, block_ids, target: str, stream: int):
        for bid in block_ids:
            bid = int(bid)
            if native_available():
                advise = _native.advise_grace if target == "grace" \
                    else _native.advise_hbm
                for ptr, nbytes in self._block_ranges(layer, bid):
                    rc = advise(ptr, nbytes, stream)
                    if rc != 0:
                        self.stats["hint_errors"] += 1
            self._state[(layer, bid)] = target

    def to_grace(self, layer: str, block_ids, stream: int = 0):
        """Demote: prefer+prefetch these blocks onto Grace LPDDR5X. No copy."""
        ids = [int(b) for b in block_ids]
        if not ids:
            return
        self._apply(layer, ids, "grace", stream)
        self.stats["to_grace"] += len(ids)

    def to_hbm(self, layer: str, block_ids, stream: int = 0):
        """Restore: prefer+prefetch these blocks back onto Hopper HBM3."""
        ids = [int(b) for b in block_ids]
        if not ids:
            return
        self._apply(layer, ids, "hbm", stream)
        self.stats["to_hbm"] += len(ids)

    # -- introspection ------------------------------------------------- #
    def residency_of(self, layer: str, block_id: int) -> str:
        """Intended residency from our state machine ("grace" | "hbm")."""
        return self._state.get((layer, int(block_id)), "hbm")

    def actual_node(self, layer: str, block_id: int) -> int | None:
        """Real last-prefetch node from CUDA (device ordinal, -1==Grace).

        ``None`` off-hardware. This is the ground truth used to confirm a
        demotion really moved pages (the Seer "is it actually bound to the
        hot path" check).
        """
        if not native_available():
            return None
        ranges = self._block_ranges(layer, int(block_id))
        ptr, nbytes = ranges[0]
        return int(_native.query_node(ptr, nbytes))

    def footprint(self) -> dict:
        grace_blocks = sum(1 for v in self._state.values() if v == "grace")
        # bytes resident on Grace, summed with each layer's per-block size
        grace_bytes = 0
        for (layer, _bid), node in self._state.items():
            if node == "grace":
                grace_bytes += self._layer_block_nbytes.get(layer, 0)
        total_bytes = sum(
            self._layer_block_nbytes.get(l, 0) * n
            for l, n in self._n_blocks.items())
        return {
            "managed": native_available(),
            "regime": regime(),
            "device": device_name(),
            "layers": len(self._kv),
            "grace_blocks": grace_blocks,
            "grace_bytes": grace_bytes,
            "grace_gib": grace_bytes / float(1 << 30),
            "total_kv_bytes": total_bytes,
            "total_kv_gib": total_bytes / float(1 << 30),
            "grace_resident_frac": (grace_bytes / total_bytes)
            if total_bytes else 0.0,
            **self.stats,
        }

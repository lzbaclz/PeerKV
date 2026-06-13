"""Allocator-layer wiring: land vLLM's KV pool on managed (Grace+HBM) memory.

This is the seam the ``KVConnectorBase_V1`` interface can't reach. vLLM
allocates its paged KV cache deep in the worker (a ``torch.zeros`` of the
backend's ``get_kv_cache_shape``), so to make those blocks migratable between
HBM and Grace we route *that one allocation* through a
:class:`torch.cuda.MemPool` bound to :class:`umallm.uma_alloc.UMAManagedAllocator`.
Everything else (weights, activations) keeps the default caching allocator
and stays HBM-fast.

Two ways to use it, earliest-wins (must run before the KV cache is built):

1. Auto patch -- call once at process start (e.g. a vLLM plugin entrypoint or
   before constructing ``LLM``)::

       from umallm.vllm_integration.uma_backend import patch_vllm_kv_allocation
       patch_vllm_kv_allocation()

2. Manual scope -- if you own the allocation site::

       with kv_alloc_scope():
           kv_cache = torch.zeros(shape, ...)

Off-CUDA this module imports fine and every entry point degrades to a no-op
so the connector still loads in a CPU sandbox.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import logging

from ..uma_alloc import UMAManagedAllocator, UMAResidencyController, native_available

logger = logging.getLogger(__name__)

_POOL = None  # cached torch.cuda.MemPool (one managed pool per process)
_ALLOC = None  # retain the UMAManagedAllocator so its CUDAPluggableAllocator
# (which owns the malloc/free fn pointers the MemPool calls into) is not GC'd;
# dropping it leaves the pool with a dangling allocator -> segfault on alloc.

# vLLM V1 KV-allocation sites across releases (module:Class.method). We wrap
# the first that exists; the shape/layout itself is untouched.
_DEFAULT_KV_ALLOC_TARGETS = (
    "vllm.v1.worker.gpu_model_runner:GPUModelRunner.initialize_kv_cache",
    "vllm.v1.worker.gpu_worker:Worker.initialize_cache",
    "vllm.worker.worker:Worker.initialize_cache",
)


def uma_mem_pool():
    """The process-wide managed MemPool (created lazily). ``None`` off-CUDA."""
    global _POOL, _ALLOC
    if _POOL is not None:
        return _POOL
    if not native_available():
        return None
    _ALLOC = UMAManagedAllocator()          # keep alive for the process lifetime
    _POOL = _ALLOC.mem_pool()
    return _POOL


@contextlib.contextmanager
def kv_alloc_scope():
    """Allocate KV tensors inside this scope to place them on managed memory."""
    pool = uma_mem_pool()
    if pool is None:
        yield
        return
    import torch

    with torch.cuda.use_mem_pool(pool):
        yield


def _resolve(spec: str):
    modpath, _, qual = spec.partition(":")
    mod = importlib.import_module(modpath)
    owner = mod
    parts = qual.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return owner, parts[-1]


def patch_vllm_kv_allocation(target: str | None = None, pool=None) -> bool:
    """Wrap vLLM's KV-cache allocation so it draws from the managed pool.

    Returns True if a target was patched. Best-effort and idempotent: tries
    known V1 allocation sites (override with ``target="mod:Class.method"``).
    Must be called before the engine builds its KV cache.
    """
    if not native_available():
        logger.info("UMA: native managed allocator unavailable; KV stays on "
                    "the default HBM allocator (Route A fallback).")
        return False
    pool = pool or uma_mem_pool()
    candidates = (target,) if target else _DEFAULT_KV_ALLOC_TARGETS
    import torch

    for spec in candidates:
        if not spec:
            continue
        try:
            owner, name = _resolve(spec)
            orig = getattr(owner, name)
        except (ImportError, AttributeError):
            continue
        if getattr(orig, "_uma_wrapped", False):
            return True

        @functools.wraps(orig)
        def wrapper(*args, __orig=orig, **kwargs):
            with torch.cuda.use_mem_pool(pool):
                return __orig(*args, **kwargs)

        wrapper._uma_wrapped = True
        setattr(owner, name, wrapper)
        logger.info("UMA: KV allocation routed through managed MemPool via %s",
                    spec)
        return True

    logger.warning(
        "UMA: no known vLLM KV-allocation site found to patch; allocate KV "
        "inside uma_backend.kv_alloc_scope() instead.")
    return False


def make_residency_controller(block_dim: int = 1) -> UMAResidencyController:
    """Controller for the FlashAttention V1 ``(2, num_blocks, ...)`` layout.

    ``block_dim`` defaults to 1 (the K/V axis is outermost); pass 0 for a
    ``(num_blocks, ...)`` layout.
    """
    return UMAResidencyController(block_dim=block_dim)

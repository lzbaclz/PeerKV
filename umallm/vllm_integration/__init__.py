"""vLLM integration for UMA-LLM on NVIDIA Grace-Hopper (GH200).

GH200 exposes a *coherent* unified memory: the Grace CPU's LPDDR5X is
visible to the Hopper GPU over NVLink-C2C at ~900 GB/s with no PCIe hop.
That makes vLLM (CUDA) the natural engine for the GH200 line, and it makes
the "cold tier" a Grace-resident, coherently-accessible buffer rather than
a PCIe-staged DRAM copy. This package plugs UMA-LLM's cost model + residency
policy into vLLM's ``KVConnectorBase_V1`` so cold KV blocks are demoted to
Grace memory (cheap, coherent) and/or KIVI-compressed in place.

Two integration depths:
  * Route A (connector): demote cold KV blocks via the ``KVConnectorBase_V1``
    interface -- a host-staging copy (offload-style). Works on any vLLM, but
    the "tier move" is a copy.
  * Route B (allocator): land the KV pool on managed (Grace+HBM) memory so a
    "tier move" is a zero-copy residency hint (``cudaMemAdvise`` +
    ``cudaMemPrefetchAsync``). This is what makes UMA-LLM structurally
    different from an offload connector and is the paper's headline claim.

Modules:
  * :mod:`placement` -- scheduler-side decision logic (pure Python / NumPy;
    cost model + sizing inversion + :class:`umallm.policy.UMAPolicy`). Fully
    testable without vLLM or CUDA.
  * :mod:`gh200_connector` -- the vLLM ``KVConnectorBase_V1`` glue (scheduler
    + worker). The worker delegates T0<->T1 to zero-copy residency hints
    (Route B) when a managed KV pool is live, else host-copy (Route A).
  * :mod:`uma_backend` -- Route B allocator wiring: route vLLM's KV
    allocation through a managed :class:`torch.cuda.MemPool`
    (:func:`patch_vllm_kv_allocation`) so blocks can migrate HBM<->Grace.

All imports of ``vllm``/``torch``/CUDA are guarded so the package loads in a
CPU sandbox; the scheduler + residency *logic* is unit-tested there. The
on-GH200 device path (managed allocation + prefetch timings vs the
calibration model) is the remaining hardware-side validation.
"""

from .placement import GH200PlacementConfig, GraceHopperPlacement, TIER_NAMES
from .uma_backend import (
    kv_alloc_scope,
    make_residency_controller,
    patch_vllm_kv_allocation,
    uma_mem_pool,
)

__all__ = [
    "GH200PlacementConfig",
    "GraceHopperPlacement",
    "TIER_NAMES",
    "kv_alloc_scope",
    "make_residency_controller",
    "patch_vllm_kv_allocation",
    "uma_mem_pool",
]

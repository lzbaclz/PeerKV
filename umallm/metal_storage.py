"""Metal storage-mode mapping — addresses round-1 M2.

On Apple Silicon, Metal exposes several storage modes for memory buffers:
- shared:    CPU+GPU coherent; both read/write at SoC bandwidth.
- private:   GPU-only; CPU access requires a blit encoder.
- managed:   CPU+GPU each have a copy; synced on commit/sync.
- memoryless: GPU-only, transient (not for KV cache).

UMA-LLM picks the right storage mode per residency tier:
  T0 GPU-active:  shared (so the policy can read tier metadata cheaply)
  T1 CPU-active:  shared (same; we let the CPU touch it freely)
  T2 compressed:  shared (the compressed bytes live in a side array)
  T3 swapped:     not a Metal buffer; managed by OS swap.

We keep `shared` across all tiers because UMA-LLM's whole point is to
exploit unified memory; the choice of mode is more about access
permissions than location.
"""
from __future__ import annotations

import enum

from .uma_model import ResidencyTier


class MetalStorageMode(enum.Enum):
    """Subset of MTLResourceStorageMode actually used by UMA-LLM."""
    SHARED = "shared"          # CPU+GPU coherent
    PRIVATE = "private"        # GPU-only (would require blit for CPU read)
    MANAGED = "managed"        # explicit sync; not used here


def storage_mode_for(tier: ResidencyTier) -> MetalStorageMode:
    """Pick storage mode for a tier.

    Note (round-1 M2 response): We deliberately use SHARED across all
    active tiers (T0/T1/T2) so the policy and the pressure listener can
    read/write tier metadata from CPU without blits. The cost of SHARED
    over PRIVATE on M2 Max is ~3% bandwidth in our probe; we trade that
    for control-plane simplicity.
    """
    if tier == ResidencyTier.T0_GPU_ACTIVE:
        return MetalStorageMode.SHARED
    if tier == ResidencyTier.T1_CPU_ACTIVE:
        return MetalStorageMode.SHARED
    if tier == ResidencyTier.T2_COMPRESSED:
        return MetalStorageMode.SHARED
    # T3 is OS swap; not a Metal buffer
    raise ValueError(f"tier {tier} is not a Metal buffer (likely T3 swap)")

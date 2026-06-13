"""Iteration 5 — Grace-Hopper GH200 cost-model variant.

GH200 differs from M2 Max in three ways:
1. SoC bandwidth: 900 GB/s (~2.2× higher) via NVLink-C2C-coherent CPU.
2. No NVMe swap tier — typically the host has DDR5 only, swap is rare.
3. CPU has its own LPDDR5X pool (480 GB) accessible from GPU coherently.

We model this with a UMACostModel subclass.
"""
from __future__ import annotations

from dataclasses import dataclass

from .uma_model import ResidencyTier, UMACostModel


@dataclass
class GraceHopperCostModel(UMACostModel):
    """GH200-specific cost-model parameters."""
    soc_bw_gbps: float = 900.0          # SoC BW including LPDDR5X coherency
    l2_miss_ns: float = 60.0             # Hopper L2 is faster
    l2_line_bytes: int = 128
    compress_us_per_kb: float = 0.8      # H100 INT4 quant is faster
    decompress_us_per_kb: float = 0.8
    swap_in_us_per_kb: float = 50.0      # GH200 NVMe rare; treat as
                                          # cluster-side restore via NVLink
    swap_out_us_per_kb: float = 20.0
    block_bytes: int = 32768

    def access_cost(self, tier: ResidencyTier,
                    block_bytes: int | None = None) -> float:
        """Override: T1 (CPU-active) on GH200 is actually fast — LPDDR5X
        is coherent with GPU via NVLink-C2C. We use the same SoC BW
        instead of CPU bandwidth derate."""
        b = block_bytes or self.block_bytes
        kb = b / 1024.0
        if tier == ResidencyTier.T0_GPU_ACTIVE:
            return b / (self.soc_bw_gbps * 1e3)
        if tier == ResidencyTier.T1_CPU_ACTIVE:
            return b / (self.soc_bw_gbps * 1e3)  # equal!
        if tier == ResidencyTier.T2_COMPRESSED:
            return self.access_cost(ResidencyTier.T0_GPU_ACTIVE, b) + \
                   kb * self.decompress_us_per_kb
        if tier == ResidencyTier.T3_SWAPPED:
            return kb * self.swap_in_us_per_kb
        raise ValueError(f"unknown tier {tier}")

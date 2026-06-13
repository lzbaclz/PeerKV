"""Scheduler-side placement logic for the GH200 vLLM connector.

Pure Python / NumPy -- no vLLM, no torch -- so it is fully unit-testable on
CPU. Given a set of KV blocks with per-block hotness and an operator SLO,
it (1) sizes the HBM-resident budget via the closed-form schedulability
inversion on the Grace-Hopper cost model, then (2) assigns each block a
residency tier via :class:`umallm.policy.UMAPolicy`.

GH200 tier mapping (vs. the discrete-memory T0..T3):
  * T0 GPU-active  -> ``hbm``         (Hopper HBM3)
  * T1 CPU-active  -> ``grace``       (Grace LPDDR5X, coherent over C2C)
  * T2 compressed  -> ``compressed``  (KIVI 4-bit, in place)
  * T3 swapped     -> folded into ``grace`` (GH200 servers have no NVMe
                      swap tier; ``has_swap=False``)

The coherence of C2C is what makes ``grace`` cheap: a block demoted there is
still read by the GPU without an explicit copy-back, unlike a PCIe-staged
host buffer -- which is exactly why the GH200 cost model sets
``access(T1) == access(T0)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..grace_hopper import GraceHopperCostModel
from ..policy import UMAPolicy
from ..pressure import PressureLevel
from ..uma_model import ResidencyTier, SizingResult

# Residency-tier int -> GH200 physical tier name. T3 folds into grace
# because GH200 has no local NVMe swap tier.
TIER_NAMES = {
    int(ResidencyTier.T0_GPU_ACTIVE): "hbm",
    int(ResidencyTier.T0_ANE_ACTIVE): "hbm",
    int(ResidencyTier.T1_CPU_ACTIVE): "grace",
    int(ResidencyTier.T2_COMPRESSED): "compressed",
    int(ResidencyTier.T3_SWAPPED): "grace",
}


@dataclass
class GH200PlacementConfig:
    """Operator-facing SLO + workload parameters."""

    deadline_ms: float = 50.0          # per-token deadline D
    miss_target: float = 1e-2          # rho
    block_size_tokens: int = 16        # vLLM block size
    ema_attention_lat_us: float = 1500.0
    ema_compute_lat_us: float = 8000.0
    gpu_frac: float = 0.8              # of the HBM budget, fraction pinned to T0
    n_sink: int = 1                    # leading blocks always resident
    n_window: int = 2                  # trailing (recent) blocks always resident
    bound_mode: str = "bernstein"      # heavy-tail safe by default


@dataclass
class GraceHopperPlacement:
    """Turns per-block hotness + SLO into a vLLM tier assignment."""

    cfg: GH200PlacementConfig = field(default_factory=GH200PlacementConfig)
    cost_model: GraceHopperCostModel = field(default_factory=GraceHopperCostModel)

    def hbm_budget(self, n_blocks: int) -> SizingResult:
        """Min HBM-resident blocks to meet (deadline, miss target) on GH200."""
        return self.cost_model.min_active_blocks_for_slo(
            deadline_us=self.cfg.deadline_ms * 1e3,
            ema_attention_lat=self.cfg.ema_attention_lat_us,
            ema_compute_lat=self.cfg.ema_compute_lat_us,
            n_blocks=max(1, n_blocks),
            miss_target=self.cfg.miss_target,
            mode=self.cfg.bound_mode,
        )

    def assign(self, block_ids, hotness,
               pressure: PressureLevel = PressureLevel.NORMAL):
        """Return ({block_id: tier_name}, SizingResult).

        ``hotness`` is a per-block score (e.g. EMA attention mass), aligned
        with ``block_ids``.
        """
        block_ids = list(block_ids)
        n = len(block_ids)
        sizing = self.hbm_budget(n)
        if n == 0:
            return {}, sizing
        floor = self.cfg.n_sink + self.cfg.n_window
        # Tiny request: everything stays in HBM (policy invariant needs
        # n_active >= n_sink + n_window).
        if n <= floor:
            return {int(b): "hbm" for b in block_ids}, sizing

        n_active = sizing.min_active_blocks if sizing.feasible else n
        n_active = int(min(max(n_active, floor), n))

        policy = UMAPolicy(
            cost_model=self.cost_model, n_active=n_active,
            gpu_frac=self.cfg.gpu_frac, n_sink=self.cfg.n_sink,
            n_window=self.cfg.n_window,
            swap_under_critical=False,  # GH200: no NVMe swap tier
        )
        tiers = policy.place(np.asarray(hotness, dtype=np.float32), pressure=pressure)
        return ({int(b): TIER_NAMES[int(t)] for b, t in zip(block_ids, tiers)},
                sizing)

    def summarize(self, assignment: dict) -> dict:
        out = {"hbm": 0, "grace": 0, "compressed": 0}
        for name in assignment.values():
            out[name] = out.get(name, 0) + 1
        out["n_blocks"] = len(assignment)
        out["resident_hbm_frac"] = (out["hbm"] / len(assignment)) if assignment else 1.0
        return out

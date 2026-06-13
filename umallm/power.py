"""Power / energy model for UMA-LLM — addresses 100-round R05.

Provides a J/token estimator using per-tier active energy + idle/leakage.
Calibration scalars (W) are platform-specific:
  - M2 Max GPU active:   ~ 30 W
  - M2 Max CPU active:   ~ 20 W
  - M2 Max NVMe active:  ~ 4 W
  - M2 Max idle:         ~ 5 W
"""
from __future__ import annotations

from dataclasses import dataclass

from .uma_model import ResidencyTier


@dataclass
class PowerModel:
    """Per-tier active power in Watts; calibrated per device.

    On M-series Macs, `powermetrics --samplers cpu_power,gpu_power` gives
    these numbers in real time. We expose a static estimate that the
    runtime samples to compute J/token.
    """
    p_gpu_active_w: float = 30.0
    p_cpu_active_w: float = 20.0
    p_nvme_active_w: float = 4.0
    p_idle_w: float = 5.0
    # On Grace-Hopper / data-center, override these:
    p_hbm_static_w: float = 0.0   # included in GPU active for M-series

    def per_step_energy_j(self,
                          tier_distribution: dict[int, int],
                          step_duration_s: float) -> float:
        """Estimate energy (J) consumed in one decode step given the per-tier
        block count and the step's wall-clock duration."""
        n_t0 = tier_distribution.get(int(ResidencyTier.T0_GPU_ACTIVE), 0)
        n_t1 = tier_distribution.get(int(ResidencyTier.T1_CPU_ACTIVE), 0)
        n_t3 = tier_distribution.get(int(ResidencyTier.T3_SWAPPED), 0)
        # crude weighting: power proportional to blocks active per tier
        total = max(1, sum(tier_distribution.values()))
        frac_gpu = n_t0 / total
        frac_cpu = n_t1 / total
        frac_nvme = n_t3 / total
        avg_w = (self.p_idle_w +
                 frac_gpu * self.p_gpu_active_w +
                 frac_cpu * self.p_cpu_active_w +
                 frac_nvme * self.p_nvme_active_w)
        return avg_w * step_duration_s

    def per_token_energy_j(self,
                           tier_distribution: dict[int, int],
                           tpot_s: float) -> float:
        return self.per_step_energy_j(tier_distribution, tpot_s)

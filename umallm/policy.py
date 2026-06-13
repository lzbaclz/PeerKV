"""UMA placement policy.

The policy maps a per-block hotness score (produced by upstream HALO/XQP
predictor) to a residency-tier assignment that respects:
- Total RAM budget (avoid OS swap),
- Active GPU-compute working-set budget (avoid L2 thrash),
- Current memory-pressure level (from `pressure.MemoryPressureListener`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pressure import PressureLevel
from .uma_model import ResidencyTier, UMACostModel


@dataclass
class UMAPolicy:
    """Single-knob policy: given hotness and pressure, place each block."""
    cost_model: UMACostModel = field(default_factory=UMACostModel)
    # Active-tier budget (T0+T1) in blocks. The compression tier T2 has
    # no explicit budget — it fills the rest of the RAM up to swap line.
    n_active: int = 2048
    # T0 vs T1 split: top-X% of active blocks pinned to GPU
    gpu_frac: float = 0.8
    # Pressure response thresholds
    compress_under_warn: bool = True
    swap_under_critical: bool = True
    # Sink + window: never compressed
    n_sink: int = 4
    n_window: int = 4
    # Round-2 M1: pre-emptive headroom — if free RAM falls below this
    # fraction, demote eagerly *before* the kernel even runs. This is
    # the synchronous knob that races ahead of the asynchronous pressure
    # listener.
    headroom_frac: float = 0.10

    def __post_init__(self):
        # BUGFIX (audit): sink + window must fit within n_active or the
        # policy silently violates its own invariant.
        if self.n_sink + self.n_window > self.n_active:
            raise ValueError(
                f"n_sink ({self.n_sink}) + n_window ({self.n_window}) "
                f"must be <= n_active ({self.n_active})"
            )

    def place(self, scores: np.ndarray, pressure: PressureLevel
              = PressureLevel.NORMAL) -> np.ndarray:
        """Return a (n_blocks,) array of ResidencyTier values."""
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        n = scores.shape[0]
        tiers = np.full(n, int(ResidencyTier.T2_COMPRESSED), dtype=np.int32)

        # Force sink + window into T0
        sink_idx = list(range(min(self.n_sink, n)))
        win_idx = list(range(max(0, n - self.n_window), n))
        keep_set = set(sink_idx) | set(win_idx)
        for i in keep_set:
            tiers[i] = int(ResidencyTier.T0_GPU_ACTIVE)

        # Remaining active budget
        remaining = max(0, self.n_active - len(keep_set))
        if remaining > 0:
            # rank by score (descending), exclude already-placed
            order = np.argsort(-scores)
            picks = [int(i) for i in order if int(i) not in keep_set]
            picks = picks[:remaining]
            n_gpu = int(self.gpu_frac * len(picks))
            for j, i in enumerate(picks):
                tiers[i] = int(
                    ResidencyTier.T0_GPU_ACTIVE if j < n_gpu
                    else ResidencyTier.T1_CPU_ACTIVE
                )

        # Pressure response: under WARN, compress more (shrink T1).
        if pressure >= PressureLevel.WARN and self.compress_under_warn:
            # downgrade T1 blocks (the lower-priority active half) to T2
            t1_idx = np.where(tiers == int(ResidencyTier.T1_CPU_ACTIVE))[0]
            n_demote = len(t1_idx) // 2 if pressure == PressureLevel.WARN else len(t1_idx)
            # demote the lowest-score ones
            t1_sorted = sorted(t1_idx.tolist(), key=lambda i: float(scores[i]))
            for i in t1_sorted[:n_demote]:
                tiers[i] = int(ResidencyTier.T2_COMPRESSED)

        # Under CRITICAL, demote bottom of T2 to T3 (swap)
        if pressure == PressureLevel.CRITICAL and self.swap_under_critical:
            t2_idx = np.where(tiers == int(ResidencyTier.T2_COMPRESSED))[0]
            t2_sorted = sorted(t2_idx.tolist(), key=lambda i: float(scores[i]))
            n_swap = len(t2_sorted) // 4
            for i in t2_sorted[:n_swap]:
                tiers[i] = int(ResidencyTier.T3_SWAPPED)

        return tiers

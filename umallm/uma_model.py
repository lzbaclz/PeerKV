"""UMA cost model and schedulability inversion.

This module is the theoretical core of UMA-LLM. It replaces the PCIe
transfer model used by every prior tier-aware KV-cache system
(OrchKvCache / SEER / InfiniGen / FlexGen) with a **unified-memory** cost
model whose dominant terms are cache-miss latency and in-place
(de)compression, *not* a bus copy.

It exposes:

* :class:`ResidencyTier` — the four UMA residency states.
* :class:`DeviceClass` / :data:`KNOWN_DEVICES` — per-SoC parameters.
* :class:`UMACostModel` — ``cost`` (move) and ``access_cost`` (read), plus
  the **closed-form schedulability inversion** :meth:`min_active_blocks_for_slo`
  that, given an operator deadline ``D`` and miss target ``rho``, returns the
  minimum active-tier budget that keeps the per-step deadline-miss
  probability under ``rho``.

The inversion is the UMA analogue of SEER's Lemma 2: the per-step cost is a
sum of Bernoulli slow-tier penalties, and we upper-bound the deadline-miss
probability with a sub-Gaussian (or Bernstein, heavy-tail) tail and invert
it for the budget. The single substituted quantity relative to SEER is the
per-block slow-tier penalty ``ell_bar``, which on UMA is
``access_cost(T2) - access_cost(T0)`` (a dequant + cache-warm cost) rather
than a PCIe read.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass


class ResidencyTier(enum.IntEnum):
    """Residency states on a unified-memory system.

    These are *residency states*, not physical locations -- the data
    physically lives in one pool. The tier captures what state the data is
    in (active vs. compressed vs. swapped) and which compute unit last
    touched it (GPU vs. CPU).
    """

    T0_GPU_ACTIVE = 0
    T1_CPU_ACTIVE = 1
    T2_COMPRESSED = 2
    T3_SWAPPED = 3
    # ANE as a sibling of T0; reserved for future Neural-Engine integration.
    # Not used by the placement policy today.
    T0_ANE_ACTIVE = 4


@dataclass(frozen=True)
class DeviceClass:
    """Per-SoC parameters used to initialise the cost model.

    Different M-series chips (and Grace-Hopper) have very different SoC
    bandwidth and thermal envelopes, which changes the active-budget
    arithmetic, so we make the device explicit rather than hard-coding M2.
    """

    name: str = "M2_Max"
    soc_bw_gbps: float = 400.0
    cores_gpu: int = 38
    has_ane: bool = True
    thermal_envelope_w: float = 60.0
    has_swap: bool = True  # GH200 servers typically have no NVMe swap tier


KNOWN_DEVICES = {
    "M2_Max": DeviceClass("M2_Max", 400.0, 38, True, 60.0, True),
    "M2_Ultra": DeviceClass("M2_Ultra", 800.0, 76, True, 120.0, True),
    "M3_Max": DeviceClass("M3_Max", 400.0, 40, True, 60.0, True),
    "M3_Ultra": DeviceClass("M3_Ultra", 819.0, 80, True, 140.0, True),
    "M4_Max": DeviceClass("M4_Max", 546.0, 40, True, 60.0, True),
    "M4_Pro": DeviceClass("M4_Pro", 273.0, 20, True, 40.0, True),
    # Grace-Hopper GH200: coherent NVLink-C2C CPU+GPU, ~900 GB/s, no NVMe
    # swap tier in the typical server configuration. See grace_hopper.py for
    # the cost-model override (T1 is as fast as T0 under C2C coherence).
    "GH200": DeviceClass("GH200", 900.0, 132, False, 700.0, False),
}


@dataclass
class SizingResult:
    """Outcome of the schedulability inversion.

    Attributes
    ----------
    min_active_blocks:
        Minimum number of blocks that must stay in the fast (T0/T1) tier so
        that ``Pr(C_t > D) <= rho``. If infeasible, equals ``n_blocks``.
    max_slow_fraction:
        The largest fraction of blocks that may be demoted to the slow
        (compressed/swapped) tier while still meeting the SLO. ``0.0`` when
        infeasible.
    feasible:
        Whether *any* budget meets the SLO. ``False`` when the deadline is
        below the irreducible compute+attention floor.
    deadline_floor_us:
        The compute+attention-only latency (``mu`` at zero slow fraction).
        If ``D <= deadline_floor_us`` no budget can help -- this is the
        UMA analogue of SEER's "deadline-floor precondition".
    bound_at_solution:
        The tail-bound value ``Pr(C_t > D)`` achieved at the returned
        budget (should be ``<= rho`` when feasible).
    mode:
        ``"subgaussian"`` or ``"bernstein"``.
    """

    min_active_blocks: int
    max_slow_fraction: float
    feasible: bool
    deadline_floor_us: float
    bound_at_solution: float
    mode: str


@dataclass
class UMACostModel:
    """Cost model parameters. Defaults calibrated to Apple M2 Max.

    All latencies are in microseconds unless noted. Real deployments should
    overwrite the defaults with :func:`umallm.calibration.write_calibration`
    output measured on the target SoC.
    """

    soc_bw_gbps: float = 400.0  # SoC memory bandwidth
    l2_miss_ns: float = 80.0  # per cache line
    l2_line_bytes: int = 128
    compress_us_per_kb: float = 1.5  # KIVI 4-bit quantize on Metal
    decompress_us_per_kb: float = 1.5
    swap_in_us_per_kb: float = 10.0  # NVMe page-in on macOS
    swap_out_us_per_kb: float = 3.0
    block_bytes: int = 32768  # 32 KB default block
    cpu_bw_derate: float = 0.6  # CPU reaches ~60% of SoC BW on M-series

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_device(cls, device: str | DeviceClass) -> "UMACostModel":
        """Build a cost model seeded from a known device class."""
        dc = KNOWN_DEVICES[device] if isinstance(device, str) else device
        return cls(soc_bw_gbps=dc.soc_bw_gbps)

    @classmethod
    def from_calibration(cls, path: str) -> "UMACostModel":
        """Load probe output (see ``calibration.write_calibration``)."""
        import json

        with open(path) as fh:
            cal = json.load(fh)
        m = cls()
        if "bandwidth" in cal:
            m.soc_bw_gbps = float(cal["bandwidth"]["bandwidth_gbps"])
        if "l2_miss" in cal:
            m.l2_miss_ns = float(cal["l2_miss"]["per_line_ns"])
        if "kivi" in cal:
            m.compress_us_per_kb = float(cal["kivi"]["us_per_kb"])
            m.decompress_us_per_kb = float(cal["kivi"]["us_per_kb"])
        return m

    # ------------------------------------------------------------------ #
    # Move / access costs
    # ------------------------------------------------------------------ #
    def cost(
        self,
        from_tier: ResidencyTier,
        to_tier: ResidencyTier,
        block_bytes: int | None = None,
    ) -> float:
        """Cost (microseconds) to move a block between tiers.

        T0 <-> T1: no copy; just a memory fence + cache-warm hint.
        T0/T1 -> T2: compression (in-place; same RAM).
        T2 -> T0/T1: decompression.
        any -> T3: swap out (madvise + NVMe write-back if dirty).
        T3 -> any: page-fault + NVMe read-in.
        """
        b = block_bytes or self.block_bytes
        kb = b / 1024.0
        ft, tt = int(from_tier), int(to_tier)
        if ft == tt:
            return 0.0
        # ANE-active behaves like GPU-active for transport purposes.
        if ft == int(ResidencyTier.T0_ANE_ACTIVE):
            ft = int(ResidencyTier.T0_GPU_ACTIVE)
        if tt == int(ResidencyTier.T0_ANE_ACTIVE):
            tt = int(ResidencyTier.T0_GPU_ACTIVE)
        if ft == tt:
            return 0.0
        # T0 <-> T1 -- cache-warm cost only (touch each line at least once).
        if {ft, tt} <= {0, 1}:
            n_lines = b // self.l2_line_bytes
            return n_lines * self.l2_miss_ns / 1e3
        # Anything -> T2: compress.
        if tt == 2:
            return kb * self.compress_us_per_kb
        # T2 -> T0/T1: decompress.
        if ft == 2 and tt in (0, 1):
            return kb * self.decompress_us_per_kb
        # -> T3 swap out (cold path).
        if tt == 3:
            return kb * self.swap_out_us_per_kb
        # T3 -> other: swap in (+ recompress if landing in T2).
        if ft == 3:
            base = kb * self.swap_in_us_per_kb
            if tt == 2:
                base += kb * self.compress_us_per_kb
            return base
        raise ValueError(f"unhandled transition {from_tier} -> {to_tier}")

    def access_cost(
        self, tier: ResidencyTier, block_bytes: int | None = None
    ) -> float:
        """Cost (microseconds) to *read* a block at a given tier."""
        b = block_bytes or self.block_bytes
        kb = b / 1024.0
        if tier in (ResidencyTier.T0_GPU_ACTIVE, ResidencyTier.T0_ANE_ACTIVE):
            return b / (self.soc_bw_gbps * 1e3)  # bytes / (GB/s) -> us
        if tier == ResidencyTier.T1_CPU_ACTIVE:
            return b / (self.soc_bw_gbps * self.cpu_bw_derate * 1e3)
        if tier == ResidencyTier.T2_COMPRESSED:
            return self.access_cost(ResidencyTier.T0_GPU_ACTIVE, b) + (
                kb * self.decompress_us_per_kb
            )
        if tier == ResidencyTier.T3_SWAPPED:
            return kb * self.swap_in_us_per_kb
        raise ValueError(f"unknown tier {tier}")

    def slow_tier_penalty(
        self,
        slow_tier: ResidencyTier = ResidencyTier.T2_COMPRESSED,
        block_bytes: int | None = None,
    ) -> float:
        """Extra latency (``ell_bar``) of accessing a block from the slow
        tier rather than from T0. This is the UMA substitute for SEER's
        per-block PCIe read latency.
        """
        return self.access_cost(slow_tier, block_bytes) - self.access_cost(
            ResidencyTier.T0_GPU_ACTIVE, block_bytes
        )

    # ------------------------------------------------------------------ #
    # Schedulability inversion (the theoretical core)
    # ------------------------------------------------------------------ #
    def _mu_sigma(
        self,
        slow_fraction: float,
        ema_attention_lat: float,
        ema_compute_lat: float,
        n_blocks: int,
        ell_bar: float,
        predictor_recall: float,
        range_factor: float,
    ) -> tuple[float, float]:
        """Mean and std of per-step cost as a function of the slow fraction.

        Each block independently incurs the slow-tier penalty with effective
        probability ``eps = slow_fraction * (1 - predictor_recall)`` (a hot
        block that was demoted *and* not caught by the predictor). The
        penalty count is Binomial(n_blocks, eps); ``range_factor * ell_bar``
        is the sub-Gaussian range proxy (matches SEER's 4*ell convention by
        default).
        """
        eps = max(0.0, min(1.0, slow_fraction * (1.0 - predictor_recall)))
        mu = ema_compute_lat + ema_attention_lat + n_blocks * eps * ell_bar
        var = n_blocks * eps * (1.0 - eps) * (range_factor * ell_bar) ** 2
        return mu, math.sqrt(max(var, 0.0))

    @staticmethod
    def _tail_prob(
        deadline_us: float, mu: float, sigma: float, ell_max: float, mode: str
    ) -> float:
        """Upper bound on Pr(C_t > D)."""
        if deadline_us <= mu:
            return 1.0
        if sigma <= 0.0:
            return 0.0
        delta = deadline_us - mu
        if mode == "bernstein":
            # Bernstein / sub-exponential form, robust to heavy tails (e.g.
            # NVMe GC bursts under T3). Pr <= exp(-delta^2 / (2(sigma^2 +
            # delta*b/3))) with b the max single-block penalty.
            denom = 2.0 * (sigma * sigma + delta * ell_max / 3.0)
            return math.exp(-(delta * delta) / denom)
        # Sub-Gaussian (default).
        return math.exp(-(delta * delta) / (2.0 * sigma * sigma))

    def min_active_blocks_for_slo(
        self,
        deadline_us: float,
        ema_attention_lat: float,
        ema_compute_lat: float,
        n_blocks: int,
        miss_target: float = 1e-2,
        slow_tier: ResidencyTier = ResidencyTier.T2_COMPRESSED,
        predictor_recall: float = 0.0,
        range_factor: float = 4.0,
        mode: str = "subgaussian",
        grid: int = 1024,
    ) -> SizingResult:
        """Invert the deadline-miss bound for the active-tier budget.

        Given an operator deadline ``deadline_us`` and miss target
        ``miss_target`` (= ``rho``), return the **minimum number of blocks
        that must stay in the fast tier** so that ``Pr(C_t > D) <= rho``.

        This is the actionable output an operator wants: "to hit a 50 ms P99
        with 1% miss, keep at least N blocks resident; the remaining may be
        compressed." Larger ``predictor_recall`` (the HALO/XQP hot-set
        recall) admits a smaller budget because mis-placed-but-caught blocks
        do not pay the penalty.
        """
        ell_bar = self.slow_tier_penalty(slow_tier)
        ell_max = range_factor * ell_bar
        # Deadline floor: compute + attention alone.
        floor_mu, _ = self._mu_sigma(
            0.0, ema_attention_lat, ema_compute_lat, n_blocks, ell_bar, 1.0, range_factor
        )
        if deadline_us <= floor_mu:
            return SizingResult(
                min_active_blocks=n_blocks,
                max_slow_fraction=0.0,
                feasible=False,
                deadline_floor_us=floor_mu,
                bound_at_solution=1.0,
                mode=mode,
            )
        # Scan the slow fraction from 0 -> 1 and take the largest phi that is
        # still feasible (most aggressive compression that meets the SLO).
        best_phi = 0.0
        best_bound = 0.0
        for i in range(grid + 1):
            phi = i / grid
            mu, sigma = self._mu_sigma(
                phi, ema_attention_lat, ema_compute_lat, n_blocks, ell_bar,
                predictor_recall, range_factor,
            )
            bound = self._tail_prob(deadline_us, mu, sigma, ell_max, mode)
            if bound <= miss_target:
                best_phi, best_bound = phi, bound
            else:
                break  # bound is monotone increasing in phi
        min_active = max(1, math.ceil((1.0 - best_phi) * n_blocks))
        return SizingResult(
            min_active_blocks=min_active,
            max_slow_fraction=best_phi,
            feasible=True,
            deadline_floor_us=floor_mu,
            bound_at_solution=best_bound,
            mode=mode,
        )

    # ------------------------------------------------------------------ #
    # Backward-compatible thin wrappers (kept for existing call sites/tests)
    # ------------------------------------------------------------------ #
    def min_budget_for_slo_subgaussian(
        self,
        deadline_us: float,
        ema_attention_lat: float,
        ema_compute_lat: float,
        n_blocks: int,
        slow_tier_fraction: float = 0.10,
        confidence: float = 0.01,
        use_sub_exponential: bool = False,
    ) -> int:
        """Legacy entry point. Returns the integer active-block budget.

        ``slow_tier_fraction`` is retained for API compatibility but is no
        longer assumed -- the budget is now *solved* from the deadline and
        ``confidence`` (= miss target). The argument is used only as the
        predictor-recall-free upper hint when the solver is infeasible.
        """
        res = self.min_active_blocks_for_slo(
            deadline_us=deadline_us,
            ema_attention_lat=ema_attention_lat,
            ema_compute_lat=ema_compute_lat,
            n_blocks=n_blocks,
            miss_target=confidence,
            mode="bernstein" if use_sub_exponential else "subgaussian",
        )
        if not res.feasible:
            return n_blocks
        return res.min_active_blocks

    def min_budget_for_slo(self, *args, **kwargs) -> int:
        return self.min_budget_for_slo_subgaussian(*args, **kwargs)

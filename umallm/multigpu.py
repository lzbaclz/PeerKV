"""Multi-GPU KV-cache placement: cost model + topology-aware policy (A100/NVLink).

The discrete multi-GPU memory hierarchy this paper (iccd_a100/) targets, for the
KV cache of one request:

    T0 LOCAL_HBM    local GPU HBM             ~2 TB/s
    T1 PEER_NVLINK  peer GPU HBM over NVLink  ~600 GB/s   <- the near tier we add
    T2 HOST_PCIE    host DRAM over PCIe       ~32-64 GB/s
    T3 NVME         NVMe SSD                  ~3-7 GB/s

(any tier may additionally be quantized in place). This module is the *code*
behind the design section:

    C1  cross-tier cost model        -> MultiGPUKVModel.access_cost / slow_tier_penalty
    C2  closed-form sizing inversion -> MultiGPUKVModel.min_fast_resident_for_slo
    C3  topology-aware placement     -> topology_aware_placement
    +   a cost-model decode-step estimate for RQ2 -> estimate_decode_step_us

Pure Python/NumPy, CPU-testable. Per-tier bandwidths default to vendor specs and
are overwritten by the on-hardware calibration (experiments/e15) via
:meth:`MultiGPUKVModel.from_calibration`. The sizing math mirrors
``umallm.uma_model`` (SEER-style deadline-miss inversion); the only change is
that ``ell_bar`` comes from a *chosen spill tier's* link bandwidth, and there
are two spill targets (NVLink vs PCIe) whose penalties differ ~10x.
"""
from __future__ import annotations

import enum
import json
import math
from dataclasses import dataclass

import numpy as np


class MGTier(enum.IntEnum):
    LOCAL_HBM = 0     # T0
    PEER_NVLINK = 1   # T1 -- the horizontal near tier
    HOST_PCIE = 2     # T2
    NVME = 3          # T3


@dataclass(frozen=True)
class MultiGPUDevice:
    """Per-tier link bandwidth (GB/s) and KV-available capacity (GB)."""
    name: str
    bw_gbps: tuple   # (HBM, NVLink, PCIe, NVMe)
    cap_gb: tuple    # free capacity available to KV per tier


# A100-80GB SXM. cap_gb = *KV-available* memory per tier (after model weights).
# Vendor-PEAK reference: NVLink3 ~600 GB/s is the *bidirectional aggregate*;
# HBM2e ~2 TB/s; PCIe Gen4 ~32-64 GB/s. Kept for comparison only.
A100_NVLINK_VENDOR = MultiGPUDevice(
    name="A100-NVLink-vendor",
    bw_gbps=(2000.0, 600.0, 64.0, 5.0),
    cap_gb=(40.0, 70.0, 512.0, 4096.0),
)
# MEASURED per-tier KV-block bandwidth on the dual-A100-SXM4 NV12 box (e15/e20):
# *unidirectional* rate a single consumer sees -- the honest default so no
# analysis is silently 2.2x optimistic on the vendor 600. Override per box with
# from_calibration(e15.json). T1=273 (not 600) is the realized one-way NVLink3.
A100_NVLINK = MultiGPUDevice(
    name="A100-NVLink",
    bw_gbps=(773.0, 273.0, 24.0, 5.0),
    cap_gb=(40.0, 70.0, 512.0, 4096.0),
)
KNOWN_MULTIGPU = {"A100-NVLink": A100_NVLINK, "A100-NVLink-vendor": A100_NVLINK_VENDOR}


# MEASURED per-tier fixed setup c_i (us) -- the intercept of fetch-latency vs
# bytes (e20 calibration on the dual-A100 box). Replaces the old single
# setup_us=1.0, which under-counted the real per-transfer cost 10-22x and so
# mispredicted launch-bound (small-block) fetches. NOTE c_T1 (NVLink, ~22us) >
# c_T2 (PCIe, ~12us): a peer copy has higher fixed launch cost than a pinned
# H2D DMA, which is exactly why naive *small-block* NVLink tiering loses and
# coalescing into large transfers is required to realize NVLink's bandwidth.
A100_C_US = (12.3, 23.6, 12.6, 50.0)   # (HBM, NVLink, PCIe, NVMe), measured e20


@dataclass
class MultiGPUKVModel:
    """C1: per-tier access cost = c_i + b/beta_i (both calibrated, not vendor)."""
    device: MultiGPUDevice = A100_NVLINK
    c_us: tuple = A100_C_US  # per-tier fixed setup (us); see A100_C_US

    @property
    def setup_us(self) -> float:  # back-compat: the NVLink-tier setup
        return self.c_us[int(MGTier.PEER_NVLINK)]

    @classmethod
    def from_calibration(cls, path_or_dict, device: MultiGPUDevice = A100_NVLINK,
                         setup_calib=None):
        """Override per-tier bandwidth (and optionally setup c_i) from probes.

        ``path_or_dict``: an e15 JSON (T0_local_hbm_gbps / T1_peer_nvlink_gbps /
        T2_host_pcie_gbps) for the bandwidths. ``setup_calib``: an optional e20
        tier_calib JSON ({"tiers": {"T0_local_hbm": {"c_us", "beta_gbps"}, ...}})
        for the measured per-tier setup c_i (and, if present, beta). Either may be
        a path or dict; missing keys keep the device default.
        """
        def _load(x):
            return json.loads(open(x).read()) if isinstance(x, str) else x
        d = _load(path_or_dict)
        bw = list(device.bw_gbps)
        for i, k in [(0, "T0_local_hbm_gbps"), (1, "T1_peer_nvlink_gbps"),
                     (2, "T2_host_pcie_gbps"), (3, "T3_nvme_gbps")]:
            if d.get(k):
                bw[i] = float(d[k])
        c = list(A100_C_US)
        for src in (d, _load(setup_calib) if setup_calib else None):
            tiers = (src or {}).get("tiers") if src else None
            if not tiers:
                continue
            for i, name in [(0, "T0_local_hbm"), (1, "T1_peer_nvlink"), (2, "T2_host_pcie")]:
                t = tiers.get(name)
                if not t:
                    continue
                if t.get("beta_gbps"):
                    bw[i] = float(t["beta_gbps"])
                if t.get("c_us") is not None:
                    c[i] = float(t["c_us"])
        dev = MultiGPUDevice(device.name + "+cal", tuple(bw), device.cap_gb)
        return cls(device=dev, c_us=tuple(c))

    def access_cost(self, tier: MGTier, block_bytes: int) -> float:
        """Microseconds to read one KV block resident in ``tier``: c_i + b/beta_i."""
        bw = self.device.bw_gbps[int(tier)]
        return self.c_us[int(tier)] + block_bytes / (bw * 1e3)  # bytes/(GB/s)->us

    def slow_tier_penalty(self, spill_tier: MGTier, block_bytes: int) -> float:
        """ell_bar: extra cost of reading a block from ``spill_tier`` vs T0."""
        return (self.access_cost(spill_tier, block_bytes)
                - self.access_cost(MGTier.LOCAL_HBM, block_bytes))

    # ----- C2: closed-form schedulability inversion --------------------- #
    def min_fast_resident_for_slo(
        self, deadline_us: float, n_blocks: int, block_bytes: int,
        spill_tier: MGTier = MGTier.PEER_NVLINK, miss_target: float = 1e-2,
        compute_us: float = 0.0, attn_us: float = 0.0,
        predictor_recall: float = 0.0, range_factor: float = 4.0,
        mode: str = "subgaussian", grid: int = 1024,
    ) -> dict:
        """Minimum #blocks that must stay fast (T0/T1) to meet (D, rho).

        Mirrors umallm.uma_model's inversion but with ell_bar set by the chosen
        spill tier. Because ell_bar(NVLink) << ell_bar(PCIe), a given SLO admits
        a much larger slow fraction (longer context) when spilling over NVLink.
        """
        ell_bar = self.slow_tier_penalty(spill_tier, block_bytes)
        ell_max = range_factor * ell_bar

        def mu_sigma(phi):
            eps = max(0.0, min(1.0, phi * (1.0 - predictor_recall)))
            mu = compute_us + attn_us + n_blocks * eps * ell_bar
            var = n_blocks * eps * (1.0 - eps) * (range_factor * ell_bar) ** 2
            return mu, math.sqrt(max(var, 0.0))

        def tail(mu, sigma):
            if deadline_us <= mu:
                return 1.0
            if sigma <= 0.0:
                return 0.0
            d = deadline_us - mu
            if mode == "bernstein":
                return math.exp(-(d * d) / (2.0 * (sigma * sigma + d * ell_max / 3.0)))
            return math.exp(-(d * d) / (2.0 * sigma * sigma))

        floor_mu, _ = mu_sigma(0.0)
        if deadline_us <= floor_mu:
            return {"feasible": False, "min_fast_blocks": n_blocks,
                    "max_slow_fraction": 0.0, "ell_bar_us": ell_bar,
                    "spill_tier": MGTier(spill_tier).name,
                    "deadline_floor_us": floor_mu, "bound_at_solution": 1.0}
        best_phi, best_bound = 0.0, 0.0
        for i in range(grid + 1):
            phi = i / grid
            mu, sigma = mu_sigma(phi)
            b = tail(mu, sigma)
            if b <= miss_target:
                best_phi, best_bound = phi, b
            else:
                break  # monotone in phi
        return {"feasible": True,
                "min_fast_blocks": max(1, math.ceil((1.0 - best_phi) * n_blocks)),
                "max_slow_fraction": best_phi, "ell_bar_us": ell_bar,
                "spill_tier": MGTier(spill_tier).name,
                "deadline_floor_us": floor_mu, "bound_at_solution": best_bound}

    # ----- C2': compute-follows-KV sizing (PeerKV-Parallel) -------------- #
    def decode_step_parallel_us(
        self, blocks_per_device, block_bytes, compute_us: float = 0.0,
        part_bytes: "int | None" = None, heads: int = 0, head_dim: int = 0,
    ) -> float:
        """Compute-follows-KV decode step: each device computes the attention
        partial over ITS OWN resident KV (read from local HBM, beta_0), in
        parallel; only the KB-sized (O, lse) partial of each *remote* shard crosses
        NVLink. Step ~= compute + max_d(local read on device d) + fixed partial
        transfer -- independent of KV size on the link. Contrast
        :func:`estimate_decode_step_us` (copy-back: serial sum of remote-link reads,
        which moves the GBs of KV).
        """
        bpd = [int(n) for n in blocks_per_device if int(n) > 0]
        if not bpd:
            return compute_us
        hbm = self.device.bw_gbps[int(MGTier.LOCAL_HBM)]
        c_hbm = self.c_us[int(MGTier.LOCAL_HBM)]
        reads = [c_hbm + n * block_bytes / (hbm * 1e3) for n in bpd]
        if part_bytes is None:
            part_bytes = partial_bytes(heads, head_dim) if (heads and head_dim) else 2080
        n_remote = max(0, len(bpd) - 1)          # one shard is local to compute GPU
        nvlink = self.device.bw_gbps[int(MGTier.PEER_NVLINK)]
        transfer = n_remote * (self.c_us[int(MGTier.PEER_NVLINK)]
                               + part_bytes / (nvlink * 1e3))
        return compute_us + max(reads) + transfer

    def max_context_for_slo_parallel(
        self, deadline_us: float, block_bytes: int, n_devices: int = 2,
        compute_us: float = 0.0, attn_us: float = 0.0, part_bytes: int = 2080,
    ) -> dict:
        """Max total KV blocks decodable within ``deadline_us`` under
        compute-follows-KV: each of ``n_devices`` reads its shard from local HBM in
        parallel, so admissible context scales ~linearly in ``n_devices`` (vs the
        single-GPU C2 inversion). This is the parallel-design counterpart of
        :meth:`min_fast_resident_for_slo`.
        """
        hbm = self.device.bw_gbps[int(MGTier.LOCAL_HBM)]
        c_hbm = self.c_us[int(MGTier.LOCAL_HBM)]
        nvlink = self.device.bw_gbps[int(MGTier.PEER_NVLINK)]
        transfer = (n_devices - 1) * (self.c_us[int(MGTier.PEER_NVLINK)]
                                      + part_bytes / (nvlink * 1e3))
        budget = deadline_us - compute_us - attn_us - transfer - c_hbm
        if budget <= 0:
            return {"feasible": False, "max_blocks": 0,
                    "max_blocks_per_device": 0, "n_devices": n_devices}
        per_dev = budget * hbm * 1e3 / block_bytes
        return {"feasible": True, "max_blocks": int(n_devices * per_dev),
                "max_blocks_per_device": int(per_dev), "n_devices": n_devices,
                "speedup_vs_single": float(n_devices)}


def capacities_in_blocks(device: MultiGPUDevice, block_bytes: int) -> dict:
    """Max #KV blocks each tier can hold, from its free capacity."""
    return {t: int(device.cap_gb[t] * 1e9 // block_bytes) for t in range(4)}


def topology_aware_placement(
    scores: np.ndarray, capacities_blocks: dict,
    n_sink: int = 1, n_window: int = 8,
    tier_bw: "dict | tuple | None" = None,
) -> np.ndarray:
    """C3: assign each completed block a tier, bandwidth-greedy + topology-aware.

    ``scores[i]`` is block i's hotness (higher = hotter). Rule:
      1. sink (first ``n_sink``) and window (last ``n_window``) blocks are pinned
         to the fast path first (they must stay resident);
      2. the remaining blocks are placed fastest-tier-first by descending hotness
         within each tier's capacity.

    ``tier_bw`` makes the policy **calibration-driven and topology-robust**: when
    given (a dict {tier:gbps} or a 4-tuple, e.g. from the measured e15 bandwidths
    or ``model.device.bw_gbps``), tiers are visited in *descending measured
    bandwidth* order, so the spill always goes to the fastest reachable tier. If
    NVLink degrades below PCIe (e.g. MIG re-enabled, link untrained), this routes
    overflow to host instead of self-harming on a slow peer link. When ``tier_bw``
    is None the legacy fixed order [HBM, NVLink, PCIe, NVMe] is used (NVLink is the
    near tier, before host) -- correct only when NVLink > PCIe, which is the
    intended A100/NVLink topology.
    Returns an int array of MGTier values. Blocks past all capacity -> NVME.
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = scores.shape[0]
    tier = np.full(n, int(MGTier.NVME), dtype=np.int32)
    forced = list(range(min(n_sink, n))) + list(range(max(0, n - n_window), n))
    forced = sorted(set(forced))
    forced_set = set(forced)
    rest = [int(i) for i in np.argsort(-scores) if int(i) not in forced_set]
    seq = forced + rest  # forced get the fastest slots first

    order = [MGTier.LOCAL_HBM, MGTier.PEER_NVLINK, MGTier.HOST_PCIE, MGTier.NVME]
    if tier_bw is not None:
        bw = (lambda t: tier_bw[int(t)]) if not isinstance(tier_bw, dict) \
            else (lambda t: tier_bw.get(int(t), 0.0))
        order = sorted(order, key=lambda t: -float(bw(t)))  # fastest measured tier first
    ti, used = 0, 0
    for i in seq:
        while ti < len(order) and used >= capacities_blocks.get(int(order[ti]), 0):
            ti += 1
            used = 0
        if ti >= len(order):
            tier[i] = int(MGTier.NVME)  # overflow everything -> would OOM/swap
            continue
        tier[i] = int(order[ti])
        used += 1
    return tier


def optimal_chunk_blocks(model: MultiGPUKVModel, block_bytes: int,
                         peak_budget_blocks: int = 1024,
                         spill_tier: MGTier = MGTier.PEER_NVLINK,
                         overhead_frac: float = 0.1) -> int:
    """C*: cost-model-derived coalescing factor for the spill tier.

    Fetching one paged block at a time is launch-bound: the fixed setup c_i
    dominates and the link's bandwidth advantage is hidden (small-block NVLink
    even loses, since c_{T1} > c_{T2}). Coalescing C consecutive blocks into one
    transfer costs ``c_i + C*b/beta_i``; the setup is a fraction
    ``c_i/(c_i + C*b/beta_i)`` of the chunk. C* is the smallest C that drives
    that fraction at/below ``overhead_frac`` (so the fetch is bandwidth-bound),
    capped by the peak budget (peak holds ~C blocks). This removes the
    "works only with a hand-tuned chunk size" failure mode: the model PICKS C.
    """
    c = model.c_us[int(spill_tier)]
    beta = model.device.bw_gbps[int(spill_tier)]
    per_block_us = block_bytes / (beta * 1e3)               # bandwidth term / block
    # setup fraction <= overhead_frac  <=>  C >= c(1-f)/(f * per_block_us)
    c_star = math.ceil(c * (1.0 - overhead_frac) / (overhead_frac * per_block_us)) \
        if per_block_us > 0 else peak_budget_blocks
    return max(1, min(int(c_star), int(peak_budget_blocks)))


def estimate_decode_step_us(
    placement: np.ndarray, model: MultiGPUKVModel, block_bytes: int,
    compute_us: float = 0.0, overlap: bool = False,
) -> float:
    """Cost-model estimate (RQ2, predicted) of one decode step's KV-read time.

    Decode attends over all KV, so the step reads every block from its tier.
    ``overlap=False`` is the serial fetch-then-compute upper bound:
    ``compute_us + sum(access_cost)``. ``overlap=True`` models the one-step-ahead
    double-buffered prefetch (\\S impl) that hides the spill transfer behind
    compute -- ``max(compute_us + local_reads, spill_transfer)`` -- the
    FlexGen-style pipelined regime, applied fairly to BOTH NVLink and host spill
    so the model does not inflate the host penalty by assuming no overlap.
    """
    local = spill = 0.0
    for t in np.asarray(placement).reshape(-1):
        a = model.access_cost(MGTier(int(t)), block_bytes)
        if int(t) == int(MGTier.LOCAL_HBM):
            local += a
        else:
            spill += a
    if overlap:
        return max(compute_us + local, spill)
    return compute_us + local + spill


# ====================================================================== #
# Compute-follows-KV (PeerKV-Parallel) helpers
# ====================================================================== #
def partial_bytes(heads: int, head_dim: int, lse_bytes: int = 4,
                  o_bytes: int = 2) -> int:
    """Bytes of ONE decode-step online-softmax partial (normalized output ``O`` in
    fp16 + ``lse`` in fp32) exchanged per remote shard in the compute-follows-KV
    design -- crucially **independent of KV size**. That is the whole point: the
    PeerKV-Parallel step moves a few KB per shard, never the GBs of KV the
    copy-back design moves over the link.
    """
    return heads * head_dim * o_bytes + heads * lse_bytes


def balanced_partition(n_blocks: int, hbm_bw_gbps_per_device) -> list:
    """Split ``n_blocks`` across devices proportional to each device's HBM
    bandwidth, equalizing per-device KV-read time (minimizing the ``max_d`` that
    bounds a parallel decode step). For equal-bandwidth GPUs this is an even split
    -- the empirically optimal point (experiments/e20 split sweep).
    """
    bws = [float(b) for b in hbm_bw_gbps_per_device]
    tot = sum(bws) or 1.0
    alloc = [int(n_blocks * b / tot) for b in bws]
    r = n_blocks - sum(alloc)
    i = 0
    while r > 0:                              # hand out the remainder round-robin
        alloc[i % len(alloc)] += 1
        r -= 1
        i += 1
    return alloc

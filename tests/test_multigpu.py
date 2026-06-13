"""CPU tests for the multi-GPU KV placement design (umallm/multigpu.py):
cost model (C1), sizing inversion (C2), topology-aware placement (C3), and the
RQ2 decode-step cost estimate. No GPU needed -- this is the architecture-
independent core; the per-tier bandwidths are calibrated on hardware (e15)."""
import numpy as np
import pytest

from umallm.multigpu import (
    A100_NVLINK, MGTier, MultiGPUKVModel, capacities_in_blocks,
    estimate_decode_step_us, topology_aware_placement,
)

BLK = 256 * 1024            # 256 KiB KV block (per-block granularity)
BLK_COALESCED = 8 * 1024 * 1024   # 8 MiB coalesced transfer (paged chunk)


def test_cost_model_tier_ordering():
    # At a COALESCED transfer size (bandwidth-bound regime), the link-bandwidth
    # ordering holds and the NVLink penalty is ~10x below PCIe -- the paper's fact.
    m = MultiGPUKVModel()
    a0 = m.access_cost(MGTier.LOCAL_HBM, BLK_COALESCED)
    a1 = m.access_cost(MGTier.PEER_NVLINK, BLK_COALESCED)
    a2 = m.access_cost(MGTier.HOST_PCIE, BLK_COALESCED)
    a3 = m.access_cost(MGTier.NVME, BLK_COALESCED)
    assert a0 < a1 < a2 < a3                      # faster link -> cheaper access
    assert m.slow_tier_penalty(MGTier.LOCAL_HBM, BLK_COALESCED) == 0.0
    p1 = m.slow_tier_penalty(MGTier.PEER_NVLINK, BLK_COALESCED)
    p2 = m.slow_tier_penalty(MGTier.HOST_PCIE, BLK_COALESCED)
    assert 0 < p1 < p2
    assert p2 / p1 > 3.0                          # ~10x bandwidth -> large gap


def test_cost_model_smallblock_launch_bound_crossover():
    # The non-obvious refined finding (now in the calibrated model): at a SMALL
    # per-block granularity the measured NVLink setup c_T1 (~22us) EXCEEDS the
    # PCIe setup c_T2 (~12us), so naive per-block NVLink tiering is NOT cheaper
    # than host -- the launch-bound regime. This is exactly why C3 must coalesce
    # spill transfers; the win is a property of large transfers, not the link alone.
    m = MultiGPUKVModel()
    assert m.c_us[int(MGTier.PEER_NVLINK)] > m.c_us[int(MGTier.HOST_PCIE)]
    p1_small = m.slow_tier_penalty(MGTier.PEER_NVLINK, BLK)
    p2_small = m.slow_tier_penalty(MGTier.HOST_PCIE, BLK)
    assert p1_small >= p2_small                   # NVLink NOT better at 256KiB blocks
    # ...but coalescing flips it: NVLink penalty << PCIe at the coalesced size
    assert m.slow_tier_penalty(MGTier.PEER_NVLINK, BLK_COALESCED) \
        < m.slow_tier_penalty(MGTier.HOST_PCIE, BLK_COALESCED)


def test_from_calibration_overrides_bandwidth():
    m = MultiGPUKVModel.from_calibration(
        {"T0_local_hbm_gbps": 1800, "T1_peer_nvlink_gbps": 500,
         "T2_host_pcie_gbps": 50})
    assert abs(m.device.bw_gbps[1] - 500) < 1e-6
    assert abs(m.device.bw_gbps[2] - 50) < 1e-6


def test_placement_pins_sink_window_to_local():
    n = 40
    scores = np.arange(n, dtype=np.float32)        # newer = hotter
    caps = {0: 5, 1: 5, 2: 100, 3: 10_000}
    tier = topology_aware_placement(scores, caps, n_sink=1, n_window=2)
    assert tier[0] == int(MGTier.LOCAL_HBM)        # sink pinned local
    assert tier[n - 1] == int(MGTier.LOCAL_HBM)    # window pinned local
    assert tier[n - 2] == int(MGTier.LOCAL_HBM)


def test_placement_spills_to_nvlink_before_pcie():
    n = 40
    scores = np.arange(n, dtype=np.float32)
    caps = {0: 4, 1: 6, 2: 100, 3: 10_000}         # 4 local, 6 NVLink, rest host
    tier = topology_aware_placement(scores, caps, n_sink=1, n_window=1)
    counts = {t: int((tier == t).sum()) for t in range(4)}
    assert counts[int(MGTier.LOCAL_HBM)] == 4      # T0 filled to capacity
    assert counts[int(MGTier.PEER_NVLINK)] == 6    # then NVLink, before any host
    assert counts[int(MGTier.HOST_PCIE)] == n - 10
    # no block sent to host while NVLink still had room
    assert counts[int(MGTier.NVME)] == 0


def test_placement_bw_aware_is_topology_robust():
    # When calibrated BW says the peer/NVLink link is SLOWER than host/PCIe
    # (the prior NODE-regime / MIG-disabled-NVLink failure mode), a BW-driven
    # policy must spill to host before peer -- i.e. NOT self-harm on the slow link.
    n = 40
    scores = np.arange(n, dtype=np.float32)
    caps = {0: 4, 1: 6, 2: 100, 3: 10_000}
    # legacy fixed order: NVLink filled before host
    fixed = topology_aware_placement(scores, caps, n_sink=1, n_window=1)
    assert int((fixed == int(MGTier.PEER_NVLINK)).sum()) == 6
    # inverted bandwidths (peer 3.6 << host 25 GB/s): route around the slow peer
    inv = topology_aware_placement(scores, caps, n_sink=1, n_window=1,
                                   tier_bw={0: 770.0, 1: 3.6, 2: 25.0, 3: 5.0})
    counts = {t: int((inv == t).sum()) for t in range(4)}
    assert counts[int(MGTier.LOCAL_HBM)] == 4         # HBM still fastest, filled first
    assert counts[int(MGTier.PEER_NVLINK)] == 0       # slow peer avoided entirely
    assert counts[int(MGTier.HOST_PCIE)] == n - 4     # spill routed to the faster host
    # with the real A100 calibration (NVLink 273 > PCIe 24), spill prefers NVLink again
    good = topology_aware_placement(scores, caps, n_sink=1, n_window=1,
                                    tier_bw={0: 773.0, 1: 273.0, 2: 24.0, 3: 5.0})
    assert int((good == int(MGTier.PEER_NVLINK)).sum()) == 6


def test_optimal_chunk_blocks_avoids_launch_bound():
    # C* must coalesce enough that the NVLink fetch is bandwidth-bound and beats
    # per-block host -- i.e. the cost model PICKS a chunk size in the winning
    # region, removing the "works only with a hand-tuned C" failure mode.
    from umallm.multigpu import optimal_chunk_blocks
    m = MultiGPUKVModel()
    b16 = 8 * 16 * 128 * 2 * 2          # 16-token paged block, K+V fp16 (H=8,D=128) = 128 KiB
    cstar = optimal_chunk_blocks(m, b16, peak_budget_blocks=1024)
    assert cstar > 1                                            # must coalesce
    assert cstar <= 1024                                        # honor peak budget
    # per-block fetch (C=1) is launch-bound: NVLink not cheaper than PCIe...
    assert m.access_cost(MGTier.PEER_NVLINK, b16) >= 0.9 * m.access_cost(MGTier.HOST_PCIE, b16)
    # ...but at C*, NVLink's amortized per-block cost is well below per-block PCIe
    nvlink_amortized = m.access_cost(MGTier.PEER_NVLINK, cstar * b16) / cstar
    assert nvlink_amortized < m.access_cost(MGTier.HOST_PCIE, b16)


def test_placement_bw_greedy_full_tier_ordering():
    # C3 on an asymmetric multi-edge topology: given 4 tiers with arbitrary
    # measured bandwidths, the BW-greedy policy must fill them in strict
    # descending-bandwidth order (fastest reachable tier first), not enum order.
    # Models a >2-GPU box where a near peer, a far (switched) peer, host, and NVMe
    # all differ -- the policy makes a real ordering decision, not a 1-edge one.
    n = 30
    scores = np.arange(n, dtype=np.float32)
    # finite caps on the two fastest; the 2nd-fastest spill tier has ample room
    caps = {0: 3, 1: 5, 2: 7, 3: 10_000}
    # bandwidths deliberately NOT in tier-enum order: descending BW = T0>T3>T2>T1
    tier_bw = {0: 900.0, 1: 40.0, 2: 80.0, 3: 300.0}
    tier = topology_aware_placement(scores, caps, n_sink=1, n_window=1, tier_bw=tier_bw)
    counts = {t: int((tier == t).sum()) for t in range(4)}
    assert counts[0] == 3                  # fastest (900) filled to its cap
    assert counts[3] == n - 3              # next-fastest (300, ample cap) absorbs all spill
    assert counts[2] == 0 and counts[1] == 0   # slower tiers (80, 40) untouched while T3 had room
    # sanity: with enum-order (no tier_bw) it would WRONGLY fill T1(40) before T3
    enum = topology_aware_placement(scores, caps, n_sink=1, n_window=1)
    assert int((enum == int(MGTier.PEER_NVLINK)).sum()) == 5   # legacy fills T1 to cap


def test_capacities_in_blocks():
    caps = capacities_in_blocks(A100_NVLINK, BLK)
    assert caps[0] == int(A100_NVLINK.cap_gb[0] * 1e9 // BLK)
    assert caps[1] > 0 and caps[2] > caps[1]


def test_sizing_nvlink_admits_more_slow_than_pcie():
    m = MultiGPUKVModel()
    # coalesced spill (bandwidth-bound): NVLink admits a larger slow fraction
    common = dict(deadline_us=50_000.0, n_blocks=512, block_bytes=BLK_COALESCED,
                  compute_us=15_000.0, attn_us=2_000.0, miss_target=1e-2)
    nv = m.min_fast_resident_for_slo(spill_tier=MGTier.PEER_NVLINK, **common)
    pc = m.min_fast_resident_for_slo(spill_tier=MGTier.HOST_PCIE, **common)
    assert nv["feasible"] and pc["feasible"]
    # spilling over the faster tier -> larger admissible slow fraction
    assert nv["max_slow_fraction"] >= pc["max_slow_fraction"]
    assert nv["min_fast_blocks"] <= pc["min_fast_blocks"]
    assert nv["ell_bar_us"] < pc["ell_bar_us"]


def test_sizing_deadline_floor_infeasible():
    m = MultiGPUKVModel()
    r = m.min_fast_resident_for_slo(
        deadline_us=1.0, n_blocks=512, block_bytes=BLK,
        compute_us=15_000.0, attn_us=2_000.0)      # D below compute+attn floor
    assert r["feasible"] is False
    assert r["min_fast_blocks"] == 512


def test_tpot_nvlink_cheaper_than_host_offload():
    m = MultiGPUKVModel()
    n = 1000
    scores = np.arange(n, dtype=np.float32)
    local = 50  # only 50 blocks fit local HBM; 950 must spill
    # NVLink-tiered: spill goes to peer GPU over NVLink
    nv = topology_aware_placement(scores, {0: local, 1: 10_000, 2: 0, 3: 0},
                                  n_sink=1, n_window=4)
    # host-offload baseline: no NVLink tier, spill goes to host PCIe
    host = topology_aware_placement(scores, {0: local, 1: 0, 2: 10_000, 3: 0},
                                    n_sink=1, n_window=4)
    all_local = topology_aware_placement(scores, {0: 10_000, 1: 0, 2: 0, 3: 0},
                                         n_sink=1, n_window=4)
    # coalesced spill (bandwidth-bound regime, where the NVLink tier pays off)
    t_nv = estimate_decode_step_us(nv, m, BLK_COALESCED, compute_us=1000.0)
    t_host = estimate_decode_step_us(host, m, BLK_COALESCED, compute_us=1000.0)
    t_local = estimate_decode_step_us(all_local, m, BLK_COALESCED, compute_us=1000.0)
    assert t_local < t_nv < t_host                 # NVLink between local and host
    assert t_host / t_nv > 2.0                      # a real, large gap
    # overlap-aware estimate: prefetch hides host transfer behind compute, but the
    # huge host spill still dominates -> NVLink remains cheaper end-to-end
    o_nv = estimate_decode_step_us(nv, m, BLK_COALESCED, compute_us=1000.0, overlap=True)
    o_host = estimate_decode_step_us(host, m, BLK_COALESCED, compute_us=1000.0, overlap=True)
    assert o_nv < o_host

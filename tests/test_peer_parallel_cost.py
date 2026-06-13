"""CPU tests for the compute-follows-KV (PeerKV-Parallel) cost model additions to
umallm/multigpu.py: parallel decode-step cost (max per-device read + KB transfer),
balanced partition, and the parallel SLO sizing (capacity scales with #devices)."""
import numpy as np

from umallm.multigpu import (
    MGTier, MultiGPUKVModel, balanced_partition, estimate_decode_step_us,
    partial_bytes,
)

BLK = 256 * 1024  # 256 KiB KV block


def test_partial_bytes_independent_of_kv_size():
    # one decode partial = O(fp16) + lse(fp32), tiny and KV-size-independent
    pb = partial_bytes(heads=8, head_dim=128)
    assert pb == 8 * 128 * 2 + 8 * 4
    assert pb < 4096                       # a few KB, not GBs


def test_parallel_step_uses_max_not_sum():
    m = MultiGPUKVModel()
    # 1000 blocks split 500/500 across two GPUs: parallel reads them concurrently
    t_par = m.decode_step_parallel_us([500, 500], BLK, heads=8, head_dim=128)
    # one GPU reading all 1000 locally (the max-term lower bound check)
    t_one = m.decode_step_parallel_us([1000], BLK, heads=8, head_dim=128)
    # splitting halves the bounding read -> ~2x faster (minus fixed terms)
    assert t_par < t_one
    assert t_one / t_par > 1.5


def test_parallel_beats_copyback_for_overflow():
    m = MultiGPUKVModel()
    n = 1000
    # copy-back: 50 local, 950 spilled to peer (moved over NVLink every step)
    scores = np.arange(n, dtype=np.float32)
    from umallm.multigpu import topology_aware_placement
    place = topology_aware_placement(scores, {0: 50, 1: 10_000, 2: 0, 3: 0},
                                     n_sink=1, n_window=4)
    t_copyback = estimate_decode_step_us(place, m, BLK, compute_us=1000.0)
    # parallel: 50 stay on compute GPU, 950 resident on peer, computed there
    t_parallel = m.decode_step_parallel_us([50, 950], BLK, compute_us=1000.0,
                                           heads=8, head_dim=128)
    assert t_parallel < t_copyback         # don't move the KV -> much cheaper
    assert t_copyback / t_parallel > 2.0


def test_balanced_partition_equalizes_and_sums():
    alloc = balanced_partition(1000, [773.0, 773.0])
    assert sum(alloc) == 1000
    assert abs(alloc[0] - alloc[1]) <= 1   # equal bw -> even split
    # bandwidth-proportional split
    alloc2 = balanced_partition(900, [600.0, 300.0])
    assert sum(alloc2) == 900
    assert alloc2[0] > alloc2[1]


def test_parallel_capacity_scales_with_devices():
    m = MultiGPUKVModel()
    common = dict(deadline_us=50_000.0, block_bytes=BLK, compute_us=15_000.0,
                  attn_us=2_000.0)
    one = m.max_context_for_slo_parallel(n_devices=1, **common)
    two = m.max_context_for_slo_parallel(n_devices=2, **common)
    assert one["feasible"] and two["feasible"]
    # two GPUs read in parallel -> ~2x the admissible context
    assert two["max_blocks"] > 1.8 * one["max_blocks"]
    # per-device capacity ~equal (2-dev only loses the tiny KB partial transfer)
    assert abs(two["max_blocks_per_device"] - one["max_blocks_per_device"]) \
        < 0.01 * one["max_blocks_per_device"]

"""CPU tests for Route B: managed-pool residency control.

These exercise the residency *brain* -- block->page-range arithmetic, the
demote/restore state machine, and Grace-vs-HBM byte accounting -- in
simulation (no CUDA). On a GH200 the same calls additionally issue
cudaMemAdvise + cudaMemPrefetchAsync; here ``native_available()`` is False so
the device hint is skipped while all bookkeeping still runs.
"""
import numpy as np
import pytest

from umallm.uma_alloc import UMAResidencyController, native_available

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:
    _CUDA = False
# These cases assert the CPU-sandbox / off-hardware FALLBACK (native UMA absent,
# no managed pool, node unknown). On a real GPU box those are intentionally
# false, so the fallback path is not exercised -- skip rather than mis-assert.
_ON_HW = native_available() or _CUDA
_skip_hw = pytest.mark.skipif(
    _ON_HW, reason="asserts off-hardware fallback; native UMA / managed pool are live on this box")


def _kv(n_blocks=8, tok=16, hd=64, seed=0):
    return np.random.default_rng(seed).standard_normal(
        (n_blocks, tok, hd)).astype(np.float32)


@_skip_hw
def test_native_unavailable_in_cpu_sandbox():
    # The whole suite runs in simulation; guard that assumption.
    assert native_available() is False


def test_register_and_block_nbytes():
    c = UMAResidencyController(block_dim=0)
    kv = _kv(n_blocks=8, tok=16, hd=64)        # 16*64*4 = 4096 B per block
    c.register_kv_caches({"l0": kv})
    assert c.block_nbytes("l0") == 16 * 64 * 4


def test_block_ranges_block_dim0_contiguous():
    c = UMAResidencyController(block_dim=0)
    kv = _kv(n_blocks=8, tok=16, hd=64)
    c.register_kv_caches({"l0": kv})
    base = kv.ctypes.data
    block_b = 16 * 64 * 4
    for bid in (0, 3, 7):
        ranges = c._block_ranges("l0", bid)
        assert ranges == [(base + bid * block_b, block_b)]


def test_block_ranges_kv_split_block_dim1():
    # FlashAttention-style (2, num_blocks, block_size, heads, head_dim):
    # block_dim=1, so each block has TWO contiguous slabs (K and V).
    c = UMAResidencyController(block_dim=1)
    kv = np.zeros((2, 4, 16, 64), dtype=np.float32)
    c.register_kv_caches({"l0": kv})
    base = kv.ctypes.data
    inner = 16 * 64 * 4                         # one block payload
    kv_stride = 4 * 16 * 64 * 4                 # stride of the K/V axis
    for bid in (0, 2, 3):
        ranges = c._block_ranges("l0", bid)
        assert ranges == [
            (base + bid * inner, inner),               # K slab
            (base + kv_stride + bid * inner, inner),   # V slab
        ]


def test_demote_restore_state_machine():
    c = UMAResidencyController(block_dim=0)
    c.register_kv_caches({"l0": _kv()})
    assert c.residency_of("l0", 2) == "hbm"        # born HBM-hot
    c.to_grace("l0", [2, 5])
    assert c.residency_of("l0", 2) == "grace"
    assert c.residency_of("l0", 5) == "grace"
    assert c.residency_of("l0", 3) == "hbm"
    c.to_hbm("l0", [2])
    assert c.residency_of("l0", 2) == "hbm"        # restored
    assert c.residency_of("l0", 5) == "grace"
    assert c.stats["to_grace"] == 2 and c.stats["to_hbm"] == 1


@_skip_hw
def test_actual_node_is_none_off_hardware():
    c = UMAResidencyController(block_dim=0)
    c.register_kv_caches({"l0": _kv()})
    c.to_grace("l0", [1])
    assert c.actual_node("l0", 1) is None          # no CUDA -> unknown


def test_footprint_accounting():
    c = UMAResidencyController(block_dim=0)
    kv = _kv(n_blocks=8, tok=16, hd=64)            # 4096 B/block, 8 blocks
    c.register_kv_caches({"l0": kv})
    c.to_grace("l0", [0, 1, 2])                     # 3 of 8 blocks to Grace
    fp = c.footprint()
    assert fp["grace_blocks"] == 3
    assert fp["grace_bytes"] == 3 * 16 * 64 * 4
    assert fp["total_kv_bytes"] == 8 * 16 * 64 * 4
    assert fp["grace_resident_frac"] == pytest.approx(3 / 8)
    assert fp["managed"] is native_available()   # managed pool only when native UMA is live


def test_footprint_multi_layer():
    c = UMAResidencyController(block_dim=0)
    c.register_kv_caches({"l0": _kv(), "l1": _kv(seed=1)})
    c.to_grace("l0", [0, 1])
    c.to_grace("l1", [0])
    fp = c.footprint()
    assert fp["grace_blocks"] == 3
    assert fp["layers"] == 2


def test_bad_block_dim_raises():
    c = UMAResidencyController(block_dim=5)
    with pytest.raises(ValueError):
        c.register_kv_caches({"l0": _kv()})


# ---------------------------------------------------------------------- #
# Connector worker, Route B path (residency hints forced on in sim).
# ---------------------------------------------------------------------- #
def test_worker_residency_demote_restore():
    from umallm.vllm_integration.gh200_connector import _Worker
    w = _Worker({"residency_hints": True, "kv_block_dim": 0})
    w.register_kv_caches({"l0": _kv(n_blocks=8)})
    w.to_grace("l0", [2, 3])
    fp = w.footprint()
    assert fp["to_grace"] == 2
    assert fp["residency"]["grace_blocks"] == 2
    # No host staging buffers are allocated on the Route B path.
    assert fp["grace_blocks"] == 0
    w.restore("l0", [2])
    fp2 = w.footprint()
    assert fp2["residency"]["grace_blocks"] == 1   # block 2 back on HBM


def test_worker_route_a_fallback_when_residency_disabled():
    from umallm.vllm_integration.gh200_connector import _Worker
    w = _Worker({"residency_hints": False})
    assert w._residency is None                     # copy path selected


# ---------------------------------------------------------------------- #
# Allocator-layer backend helpers (degrade to no-ops off-CUDA).
# ---------------------------------------------------------------------- #
@_skip_hw
def test_backend_helpers_off_cuda():
    from umallm.vllm_integration import uma_backend as b
    assert b.uma_mem_pool() is None
    assert b.patch_vllm_kv_allocation() is False
    with b.kv_alloc_scope():                          # must not raise
        pass
    ctrl = b.make_residency_controller(block_dim=1)
    assert ctrl.block_dim == 1

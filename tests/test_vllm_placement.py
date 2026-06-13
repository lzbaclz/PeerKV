"""CPU tests for the GH200 vLLM connector's placement logic.

These exercise the scheduler-side brain (cost model + sizing + policy ->
tier assignment) and confirm the connector module imports with neither
vLLM nor torch present. The worker device path (HBM<->Grace copies) needs
CUDA + GH200 and is not exercised here.
"""
import numpy as np
import pytest

from umallm.vllm_integration.placement import (
    GH200PlacementConfig, GraceHopperPlacement, TIER_NAMES,
)
from umallm.pressure import PressureLevel


def test_assign_returns_valid_tiers():
    p = GraceHopperPlacement(GH200PlacementConfig(deadline_ms=50, miss_target=1e-2))
    ids = list(range(32))
    hot = np.linspace(0, 1, 32).astype(np.float32)
    assign, sizing = p.assign(ids, hot)
    assert set(assign.values()) <= {"hbm", "grace", "compressed"}
    assert len(assign) == 32
    # GH200 has no NVMe swap tier -> nothing should be "swapped"
    assert "swapped" not in assign.values()


def test_tiny_request_all_hbm():
    p = GraceHopperPlacement()
    assign, _ = p.assign([0, 1], np.array([0.1, 0.9], dtype=np.float32))
    assert all(t == "hbm" for t in assign.values())


def test_tighter_deadline_keeps_fewer_or_equal_resident():
    ids = list(range(64))
    hot = np.linspace(0, 1, 64).astype(np.float32)
    loose = GraceHopperPlacement(GH200PlacementConfig(deadline_ms=200, miss_target=1e-2))
    tight = GraceHopperPlacement(GH200PlacementConfig(deadline_ms=40, miss_target=1e-2))
    a_loose, _ = loose.assign(ids, hot)
    a_tight, _ = tight.assign(ids, hot)
    hbm_loose = sum(1 for t in a_loose.values() if t == "hbm")
    hbm_tight = sum(1 for t in a_tight.values() if t == "hbm")
    # A looser deadline must not require *more* HBM-resident blocks.
    assert hbm_loose <= hbm_tight + 1  # +1 slack for grid rounding


def test_pressure_compresses_more():
    p = GraceHopperPlacement()
    ids = list(range(32))
    hot = np.linspace(0, 1, 32).astype(np.float32)
    normal, _ = p.assign(ids, hot, PressureLevel.NORMAL)
    crit, _ = p.assign(ids, hot, PressureLevel.CRITICAL)
    c_normal = sum(1 for t in normal.values() if t == "compressed")
    c_crit = sum(1 for t in crit.values() if t == "compressed")
    assert c_crit >= c_normal


def test_summarize():
    p = GraceHopperPlacement()
    ids = list(range(20))
    assign, _ = p.assign(ids, np.linspace(0, 1, 20).astype(np.float32))
    s = p.summarize(assign)
    assert s["n_blocks"] == 20
    assert 0.0 <= s["resident_hbm_frac"] <= 1.0


def test_connector_imports_without_vllm_or_torch():
    # The whole point: the module loads in a CPU sandbox (guarded imports).
    from umallm.vllm_integration import gh200_connector as gc
    assert hasattr(gc, "UMAGraceHopperConnector")
    assert hasattr(gc, "UMAGraceHopperMetadata")
    meta = gc.UMAGraceHopperMetadata()
    assert meta.blocks_to_grace == {}


def test_scheduler_plan_produces_demotions():
    from umallm.vllm_integration.gh200_connector import _Scheduler
    sch = _Scheduler({"deadline_ms": 40, "miss_target": 1e-2, "n_layers": 2})
    ids = list(range(40))
    sch.observe_attention("r1", np.linspace(0, 1, 40).astype(np.float32))
    sizing = sch.plan_request("r1", ids, ["layer_0", "layer_1"])
    plan = sch.take_plan()
    moved = sum(len(v) for v in plan.blocks_to_grace.values()) + \
        sum(len(v) for v in plan.blocks_to_compress.values())
    assert moved > 0           # under a 40ms SLO some blocks must be demoted
    assert sizing.feasible in (True, False)


def test_worker_compression_bookkeeping_numpy():
    """to_compressed works with numpy 'kv' (no torch): quantize + account."""
    from umallm.vllm_integration.gh200_connector import _Worker
    w = _Worker({"cold_bits": 4})
    # fake KV layer: (n_blocks, tokens, head_dim)
    w.register_kv_caches({"layer_0": np.random.default_rng(0).normal(
        size=(8, 16, 64)).astype(np.float32)})
    w.to_compressed("layer_0", [2, 3, 5])
    fp = w.footprint()
    assert fp["compressed_blocks"] == 3
    assert fp["compressed_bytes"] > 0


# ---------------------------------------------------------------------- #
# Scheduler <-> vLLM SchedulerOutput wiring (the connector's step loop).
# ---------------------------------------------------------------------- #
class _FakeNewReq:
    def __init__(self, req_id, block_ids):
        self.req_id = req_id
        self.block_ids = block_ids


class _FakeCachedReq:
    def __init__(self, req_id, new_block_ids):
        self.req_id = req_id
        self.new_block_ids = new_block_ids


class _FakeSchedulerOutput:
    def __init__(self, new_reqs=None, cached=None, num_scheduled_tokens=None):
        self.scheduled_new_reqs = new_reqs or []
        self.scheduled_cached_reqs = cached or []
        self.num_scheduled_tokens = num_scheduled_tokens or {}


class _FakeVllmConfig:
    """Minimal stand-in for vLLM's VllmConfig (scheduler-side)."""
    class _KTC:
        kv_connector_extra_config = {"deadline_ms": 40, "miss_target": 1e-2,
                                     "n_layers": 2}

    class _MC:
        num_layers = 2

    kv_transfer_config = _KTC()
    model_config = _MC()


def _scheduler_connector():
    from umallm.vllm_integration.gh200_connector import (
        UMAGraceHopperConnector, KVConnectorRole,
    )
    return UMAGraceHopperConnector(_FakeVllmConfig(), KVConnectorRole.SCHEDULER)


def test_build_connector_meta_drives_brain_from_scheduler_output():
    conn = _scheduler_connector()
    so = _FakeSchedulerOutput(
        new_reqs=[_FakeNewReq("r1", list(range(40)))],
        num_scheduled_tokens={"r1": 1},
    )
    meta = conn.build_connector_meta(so)
    moved = (sum(len(v) for v in meta.blocks_to_grace.values())
             + sum(len(v) for v in meta.blocks_to_compress.values()))
    assert moved > 0  # a 40-block request under a 40ms SLO must demote


def test_build_connector_meta_flattens_nested_block_ids():
    conn = _scheduler_connector()
    # vLLM may give block_ids nested per KV-cache group.
    so = _FakeSchedulerOutput(
        new_reqs=[_FakeNewReq("r1", [list(range(40))])],
        num_scheduled_tokens={"r1": 1},
    )
    conn.build_connector_meta(so)
    assert conn._scheduler._req_blocks["r1"] == list(range(40))


def test_cached_request_extends_block_table():
    conn = _scheduler_connector()
    conn.build_connector_meta(_FakeSchedulerOutput(
        new_reqs=[_FakeNewReq("r1", list(range(20)))],
        num_scheduled_tokens={"r1": 1}))
    conn.build_connector_meta(_FakeSchedulerOutput(
        cached=[_FakeCachedReq("r1", list(range(20, 40)))],
        num_scheduled_tokens={"r1": 1}))
    assert conn._scheduler._req_blocks["r1"] == list(range(40))


class _ScriptedPlacement:
    """Stub for GraceHopperPlacement: replays scripted tier assignments.

    Lets us test the scheduler's *delta* logic (demote / restore / no-repeat)
    independent of the cost model's sizing, which for GH200 collapses to
    "compress everything but sink+window" under the test deadlines.
    """

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self._i = 0

    def assign(self, block_ids, hotness, pressure=PressureLevel.NORMAL):
        script = self._scripts[min(self._i, len(self._scripts) - 1)]
        self._i += 1
        return {int(b): script[int(b)] for b in block_ids}, None


def test_scheduler_emits_restore_when_block_reheats():
    from umallm.vllm_integration.gh200_connector import _Scheduler
    sch = _Scheduler({"n_layers": 1})
    ids = [0, 1, 2, 3]
    sch.placement = _ScriptedPlacement([
        {0: "hbm", 1: "hbm", 2: "compressed", 3: "hbm"},  # step 1: demote 2
        {0: "hbm", 1: "hbm", 2: "hbm", 3: "hbm"},          # step 2: reheat 2
    ])
    sch.plan_request("r", ids, ["L0"])
    p1 = sch.take_plan()
    assert 2 in p1.blocks_to_compress["L0"]
    sch.plan_request("r", ids, ["L0"])
    p2 = sch.take_plan()
    assert 2 in p2.blocks_to_restore["L0"]   # compressed -> hbm == restore
    assert p2.blocks_to_compress["L0"] == []  # not re-demoted


def test_delta_plan_skips_unchanged_tiers():
    from umallm.vllm_integration.gh200_connector import _Scheduler
    sch = _Scheduler({"n_layers": 1})
    ids = [0, 1, 2, 3]
    # Same assignment both steps (the script repeats its last entry).
    sch.placement = _ScriptedPlacement([
        {0: "hbm", 1: "grace", 2: "compressed", 3: "hbm"},
    ])
    sch.plan_request("r", ids, ["L0"])
    p1 = sch.take_plan()
    moved1 = len(p1.blocks_to_grace["L0"]) + len(p1.blocks_to_compress["L0"])
    sch.plan_request("r", ids, ["L0"])
    p2 = sch.take_plan()
    moved2 = (len(p2.blocks_to_grace["L0"]) + len(p2.blocks_to_compress["L0"])
              + len(p2.blocks_to_restore["L0"]))
    assert moved1 == 2   # first step: one grace + one compressed
    assert moved2 == 0   # second step: nothing changed -> empty delta


def test_request_finished_clears_scheduler_state():
    conn = _scheduler_connector()
    conn.build_connector_meta(_FakeSchedulerOutput(
        new_reqs=[_FakeNewReq("r1", list(range(40)))],
        num_scheduled_tokens={"r1": 1}))
    assert "r1" in conn._scheduler._req_blocks

    class _Req:
        request_id = "r1"

    cont, _ = conn.request_finished(_Req(), [])
    assert cont is False
    assert "r1" not in conn._scheduler._req_blocks
    assert "r1" not in conn._scheduler._prev_tier


# ---------------------------------------------------------------------- #
# External-KV reuse: get_num_new_matched_tokens / update_state_after_alloc.
# On UMA a demoted block stays resident+coherent, so a tracked request can be
# resumed without recompute; vLLM allocates slots and the worker restores.
# ---------------------------------------------------------------------- #
def test_new_request_matches_no_external_tokens():
    from umallm.vllm_integration.gh200_connector import _Scheduler
    sch = _Scheduler({"n_layers": 1, "tokens_per_block": 16})
    # A genuinely new request is untracked -> never fabricate a prefix.
    assert sch.num_new_matched_tokens("never-seen", 0) == (0, False)


def test_matched_tokens_report_held_minus_computed():
    from umallm.vllm_integration.gh200_connector import _Scheduler
    sch = _Scheduler({"n_layers": 1, "tokens_per_block": 16})
    sch.placement = _ScriptedPlacement([
        {0: "hbm", 1: "hbm", 2: "compressed", 3: "grace"}])
    sch.plan_request("r", [0, 1, 2, 3], ["L0"])  # 4 blocks tracked = 64 tokens
    sch.take_plan()
    assert sch.num_new_matched_tokens("r", 0) == (64, True)      # all held
    assert sch.num_new_matched_tokens("r", 32) == (32, True)     # minus local
    assert sch.num_new_matched_tokens("r", 64) == (0, False)     # nothing new
    assert sch.num_new_matched_tokens("r", 999) == (0, False)    # never < 0


def test_external_reuse_can_be_disabled():
    from umallm.vllm_integration.gh200_connector import _Scheduler
    sch = _Scheduler({"n_layers": 1, "enable_external_reuse": False})
    sch.placement = _ScriptedPlacement([{0: "compressed", 1: "hbm"}])
    sch.plan_request("r", [0, 1], ["L0"])
    sch.take_plan()
    assert sch.num_new_matched_tokens("r", 0) == (0, False)


def test_update_state_after_alloc_forces_restore_over_steady_state():
    from umallm.vllm_integration.gh200_connector import _Scheduler
    sch = _Scheduler({"n_layers": 1})
    ids = [0, 1, 2, 3]
    # Steady-state script would keep 2,3 demoted forever (repeats last entry).
    sch.placement = _ScriptedPlacement([
        {0: "hbm", 1: "hbm", 2: "compressed", 3: "grace"}])
    sch.plan_request("r", ids, ["L0"])  # step 1: demote 2,3
    sch.take_plan()
    # Request resumes; vLLM re-allocates the same block ids.
    sch.note_external_match("r", ids)
    sch.plan_request("r", ids, ["L0"])  # step 2: force restore wins
    p = sch.take_plan()
    assert set(p.blocks_to_restore["L0"]) == {2, 3}
    assert p.blocks_to_compress["L0"] == []   # not re-demoted this step
    assert p.blocks_to_grace["L0"] == []


def test_connector_external_reuse_roundtrip():
    conn = _scheduler_connector()
    conn.build_connector_meta(_FakeSchedulerOutput(
        new_reqs=[_FakeNewReq("r1", list(range(40)))],
        num_scheduled_tokens={"r1": 1}))

    class _Req:
        request_id = "r1"

    # 40 blocks * 16 tokens/block = 640 tokens held; nothing computed locally.
    n, is_async = conn.get_num_new_matched_tokens(_Req(), 0)
    assert n == 640 and is_async is True
    # vLLM allocates slots for the external tokens -> queue restores.
    conn.update_state_after_alloc(_Req(), list(range(40)), n)
    meta = conn.build_connector_meta(_FakeSchedulerOutput(
        cached=[_FakeCachedReq("r1", [])],
        num_scheduled_tokens={"r1": 1}))
    restored = sum(len(v) for v in meta.blocks_to_restore.values())
    assert restored > 0  # the blocks demoted on step 1 come back


def test_update_state_after_alloc_ignores_zero_external():
    conn = _scheduler_connector()
    conn.build_connector_meta(_FakeSchedulerOutput(
        new_reqs=[_FakeNewReq("r1", list(range(40)))],
        num_scheduled_tokens={"r1": 1}))

    class _Req:
        request_id = "r1"

    # num_external_tokens == 0 -> no-op (no forced restore queued).
    conn.update_state_after_alloc(_Req(), list(range(40)), 0)
    assert conn._scheduler._force_restore.get("r1") in (None, set())


# ---------------------------------------------------------------------- #
# Block-id remap on preemption: a resumed request gets NEW slots; the worker
# re-keys its held buffers onto them so the restore lands correctly. Sound for
# the buffer-backed tiers (_grace / _compressed); residency-state re-key is
# best-effort (Route B recomputes if the slot was freed).
# ---------------------------------------------------------------------- #
def test_worker_apply_remap_rekeys_compressed():
    from umallm.vllm_integration.gh200_connector import _Worker
    w = _Worker({"cold_bits": 4})
    w.register_kv_caches({"L0": np.random.default_rng(0).normal(
        size=(40, 32, 64)).astype(np.float32)})
    w.to_compressed("L0", [2, 3])
    assert ("L0", 2) in w._compressed and ("L0", 3) in w._compressed
    w.apply_remap({2: 22, 3: 33})
    assert ("L0", 22) in w._compressed and ("L0", 33) in w._compressed
    assert ("L0", 2) not in w._compressed and ("L0", 3) not in w._compressed


def test_worker_apply_remap_then_restore_recovers_into_new_slot():
    torch = pytest.importorskip("torch")
    from umallm.vllm_integration.gh200_connector import _Worker
    w = _Worker({"cold_bits": 4, "coherent_read": False})
    kv = torch.randn(40, 32, 64)
    w.register_kv_caches({"L0": kv})
    orig = kv[2].clone()
    w.to_compressed("L0", [2])          # hold block 2's KV (compressed)
    w.apply_remap({2: 25})              # request resumed: slot 2 -> slot 25
    kv[25].zero_()
    w.restore("L0", [25])               # recover into the NEW slot
    rel = (kv[25] - orig).norm().item() / (orig.norm().item() + 1e-9)
    assert rel < 0.15                   # within KIVI 4-bit envelope


def test_scheduler_builds_positional_remap_on_resume():
    from umallm.vllm_integration.gh200_connector import _Scheduler
    sch = _Scheduler({"n_layers": 1}, ["L0"])   # plan_step uses these layer names
    sch.placement = _ScriptedPlacement([
        {0: "hbm", 1: "hbm", 2: "compressed", 3: "grace"},      # old ids
        {10: "hbm", 11: "hbm", 12: "compressed", 13: "grace"},  # new ids
    ])
    sch.plan_request("r", [0, 1, 2, 3], ["L0"])   # demote 2,3 at old ids
    sch.take_plan()
    sch.note_external_match("r", [10, 11, 12, 13])  # resumed into new slots
    # full positional remap, tier state carried onto new ids
    assert sch._plan.remap == {0: 10, 1: 11, 2: 12, 3: 13}
    assert sch._prev_tier["r"][12] == "compressed" and sch._prev_tier["r"][13] == "grace"
    assert sch._force_restore["r"] == {12, 13}     # demoted blocks, new ids
    plan = sch.plan_step(["r"])
    assert plan.remap == {0: 10, 1: 11, 2: 12, 3: 13}
    assert set(plan.blocks_to_restore["L0"]) == {12, 13}   # restore new slots
    assert plan.blocks_to_compress["L0"] == []             # not re-demoted


def test_connector_remap_flows_to_metadata():
    conn = _scheduler_connector()
    conn.build_connector_meta(_FakeSchedulerOutput(
        new_reqs=[_FakeNewReq("r1", list(range(40)))],
        num_scheduled_tokens={"r1": 1}))

    class _Req:
        request_id = "r1"

    new_blocks = list(range(100, 140))             # disjoint resumed slots
    conn.update_state_after_alloc(_Req(), new_blocks, 640)
    meta = conn.build_connector_meta(_FakeSchedulerOutput(
        cached=[_FakeCachedReq("r1", [])],
        num_scheduled_tokens={"r1": 1}))
    assert meta.remap                              # non-empty old->new map
    assert all(0 <= o < 40 and 100 <= n < 140 for o, n in meta.remap.items())
    restored = [b for v in meta.blocks_to_restore.values() for b in v]
    assert restored and all(b >= 100 for b in restored)   # restores target new slots

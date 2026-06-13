"""Unit tests for the KV placement state machine (umallm/runtime/kv_manager.py)."""
from __future__ import annotations

import threading

import pytest

from umallm.elastic_policy import PeerState
from umallm.runtime.kv_manager import (
    BlockState, IllegalTransition, KVPlacementStateMachine)

MB = 1024 ** 2


def _sm():
    sm = KVPlacementStateMachine()
    sm.register(1, 64 * MB, owner="req-a")
    sm.register(2, 64 * MB, owner="req-a")
    sm.register(3, 64 * MB, owner="req-b")
    return sm


def test_legal_transitions_and_accounting():
    sm = _sm()
    sm.migrate(1, BlockState.PEER)
    sm.migrate(2, BlockState.HOST)
    tiers = sm.bytes_by_tier()
    assert tiers["peer"] == 64 * MB
    assert tiers["host"] == 64 * MB
    assert tiers["local"] == 64 * MB
    sm.migrate(1, BlockState.LOCAL)            # restore
    assert sm.bytes_by_tier()["local"] == 128 * MB
    assert len(sm.events) == 3


def test_illegal_transitions_raise():
    sm = _sm()
    sm.migrate(1, BlockState.EVICTED)
    with pytest.raises(IllegalTransition):
        sm.migrate(1, BlockState.LOCAL)        # evicted is terminal
    sm.migrate(2, BlockState.HOST)
    with pytest.raises(IllegalTransition):
        sm.migrate(2, BlockState.PEER)         # host -> peer not in the table


def test_pinned_block_only_evictable():
    sm = KVPlacementStateMachine()
    sm.register(9, MB, migratable=False)
    with pytest.raises(IllegalTransition, match="pinned"):
        sm.migrate(9, BlockState.PEER)
    sm.migrate(9, BlockState.EVICTED)          # eviction always allowed


def test_fallback_chain_peer_unavailable():
    sm = _sm()
    full_peer = PeerState(hbm_free_bytes=0)
    assert sm.plan_migration(1, BlockState.PEER, full_peer) is BlockState.HOST
    degraded = PeerState(nvlink_bw_gbps=10.0)  # < PCIe
    assert sm.plan_migration(1, BlockState.PEER, degraded) is BlockState.HOST
    healthy = PeerState()
    assert sm.plan_migration(1, BlockState.PEER, healthy) is BlockState.PEER
    # restore from peer over a degraded link detours via host
    sm.migrate(1, BlockState.PEER)
    assert sm.plan_migration(1, BlockState.LOCAL, degraded) is BlockState.HOST


def test_release_and_thread_safety():
    sm = _sm()
    errs = []

    def worker(bid):
        try:
            for tgt in (BlockState.PEER, BlockState.LOCAL) * 50:
                sm.migrate(bid, tgt)
        except Exception as e:                  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=worker, args=(b,)) for b in (1, 2, 3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    freed = sm.release("req-a")
    assert freed == 128 * MB
    assert sm.bytes_by_tier()["evicted"] == 128 * MB

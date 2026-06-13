"""Online KV-block placement state machine (Track C P2 / Track B SS4).

Spec: docs/MASTER_PLAN.md SS3.2. Each block carries a state in
{LOCAL, PEER, HOST, COMPRESSED, EVICTED} + owner + migratable flag; migrations
are concurrency-safe (single lock, transition table); planning applies the
do-no-harm fallback chain (peer unavailable -> host; link degraded -> host).

Scope note (honest): this is the placement *state machine* -- block states,
legal transitions, byte accounting per tier, and migration planning against
live PeerState. It does NOT move tensors; the vLLM-side movers
(vllm_integration.peerkv_staging / peer_kv_alloc) execute the plans. That
separation is what makes the machine CPU-unit-testable (tests/test_runtime_kv_manager.py).
"""
from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field

from ..elastic_policy import BETA_PCIE_GBPS, PeerState

STATUS = "IMPLEMENTED: placement state machine (planning layer; movers in vllm_integration)"


class BlockState(enum.Enum):
    LOCAL = "local"
    PEER = "peer"
    HOST = "host"
    COMPRESSED = "compressed"
    EVICTED = "evicted"


# state -> states reachable in ONE legal migration
_TRANSITIONS = {
    BlockState.LOCAL:      {BlockState.PEER, BlockState.HOST,
                            BlockState.COMPRESSED, BlockState.EVICTED},
    BlockState.PEER:       {BlockState.LOCAL, BlockState.HOST, BlockState.EVICTED},
    BlockState.HOST:       {BlockState.LOCAL, BlockState.EVICTED},
    BlockState.COMPRESSED: {BlockState.LOCAL, BlockState.EVICTED},
    BlockState.EVICTED:    set(),            # terminal
}


class IllegalTransition(RuntimeError):
    """A migration not in the transition table (e.g. EVICTED -> anything)."""


@dataclass
class Block:
    block_id: int
    nbytes: int
    state: BlockState = BlockState.LOCAL
    owner: str = ""                  # request id
    migratable: bool = True
    last_moved_at: float = 0.0


@dataclass
class KVPlacementStateMachine:
    """Thread-safe placement ledger for one model replica."""
    blocks: dict = field(default_factory=dict)         # id -> Block
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _events: list = field(default_factory=list)        # (t, id, from, to)
    max_events: int = 4096

    # ---- registration -------------------------------------------------- #
    def register(self, block_id: int, nbytes: int, owner: str = "",
                 state: BlockState = BlockState.LOCAL,
                 migratable: bool = True) -> Block:
        with self._lock:
            if block_id in self.blocks:
                raise ValueError(f"block {block_id} already registered")
            b = Block(block_id, nbytes, state, owner, migratable)
            self.blocks[block_id] = b
            return b

    # ---- migration ----------------------------------------------------- #
    def migrate(self, block_id: int, target: BlockState) -> Block:
        """Commit one legal migration. Raises IllegalTransition otherwise."""
        with self._lock:
            b = self.blocks[block_id]
            if not b.migratable and target is not BlockState.EVICTED:
                raise IllegalTransition(f"block {block_id} pinned (migratable=False)")
            if target not in _TRANSITIONS[b.state]:
                raise IllegalTransition(
                    f"{b.state.value} -> {target.value} not in transition table")
            self._events.append((time.monotonic(), block_id, b.state, target))
            if len(self._events) > self.max_events:
                del self._events[:len(self._events) - self.max_events]
            b.state = target
            b.last_moved_at = time.monotonic()
            return b

    def plan_migration(self, block_id: int, want: BlockState,
                       peer: PeerState) -> BlockState:
        """Do-no-harm fallback chain: returns the tier the block should
        actually go to, given live peer state.

          want=PEER  but peer HBM full / link < PCIe  -> HOST
          want=LOCAL (restore) but link < PCIe and block on PEER -> via HOST
        Anything else passes through unchanged. Pure planning -- call
        :meth:`migrate` with the result to commit."""
        with self._lock:
            b = self.blocks[block_id]
        link_ok = peer.nvlink_bw_gbps >= BETA_PCIE_GBPS
        if want is BlockState.PEER:
            if not link_ok or peer.hbm_free_bytes < b.nbytes:
                return BlockState.HOST
        if want is BlockState.LOCAL and b.state is BlockState.PEER and not link_ok:
            return BlockState.HOST
        return want

    # ---- accounting ---------------------------------------------------- #
    def bytes_by_tier(self) -> dict:
        with self._lock:
            out = {s: 0 for s in BlockState}
            for b in self.blocks.values():
                out[b.state] += b.nbytes
            return {s.value: n for s, n in out.items()}

    def owned_by(self, owner: str) -> list:
        with self._lock:
            return [b.block_id for b in self.blocks.values() if b.owner == owner]

    def release(self, owner: str) -> int:
        """Evict every block of a finished request; returns bytes freed.
        Atomic under one lock hold: a concurrent release()/migrate() can
        neither double-count freed bytes nor record an EVICTED->EVICTED
        transition (which the transition table declares illegal)."""
        freed = 0
        with self._lock:
            for b in self.blocks.values():
                if b.owner == owner and b.state is not BlockState.EVICTED:
                    self._events.append(
                        (time.monotonic(), b.block_id, b.state, BlockState.EVICTED))
                    b.state = BlockState.EVICTED
                    freed += b.nbytes
        return freed

    @property
    def events(self) -> list:
        with self._lock:
            return list(self._events)

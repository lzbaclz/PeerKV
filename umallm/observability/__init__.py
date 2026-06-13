"""PeerKV observability (Track C area 8 / 04_cross_cutting.md SS2-SS4).

Active, runnable here:
  - ``box_probe``      -- refuse to measure on a contended shared box (9.A.4)
  - ``topology_probe`` -- startup MIG/NVLink self-check (04 SS4.3)
PLANNED stub (defined, not yet wired into the hot path):
  - ``metrics``        -- Prometheus metric set (04 SS2.2)
"""
from .box_probe import box_idle, assert_box_idle, gate_or_skip

__all__ = ["box_idle", "assert_box_idle", "gate_or_skip"]

"""Overlap-safe inter-GPU copy bandwidth measurement.

Never torch.cuda.synchronize(holder_device) during copy timing. Holder load is
kept hot via small chunked top-ups before each copy trial (holder_stream.query()).
"""
from __future__ import annotations

import statistics
from typing import Callable, Optional, Tuple

import torch

# Chunk enqueued per top-up (keeps Python enqueue fast).
HOLDER_CHUNK = 256


def topup_holder(op: Callable[[], None], holder_stream: torch.cuda.Stream) -> None:
    """Enqueue one chunk of holder work without synchronizing."""
    with torch.cuda.stream(holder_stream):
        for _ in range(HOLDER_CHUNK):
            op()


def holder_stream_busy(holder_stream: torch.cuda.Stream) -> bool:
    return not holder_stream.query()


def measure_bw_events(
    copy_fn: Callable[[], None],
    copy_stream: torch.cuda.Stream,
    nbytes: int,
    trials: int = 50,
    *,
    holder_stream: Optional[torch.cuda.Stream] = None,
    holder_op: Optional[Callable[[], None]] = None,
    assert_holder_busy: bool = False,
) -> Tuple[float, float]:
    """Return (GB/s median, relative std of event times)."""
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        with torch.cuda.stream(copy_stream):
            copy_fn()
    with torch.cuda.stream(copy_stream):
        e1.record(copy_stream)
    e1.synchronize()

    ts_ms = []
    for _ in range(trials):
        if holder_op is not None and holder_stream is not None:
            if holder_stream.query():
                topup_holder(holder_op, holder_stream)
            if assert_holder_busy and holder_stream.query():
                raise AssertionError("holder idle at copy start (overlap not established)")

        with torch.cuda.stream(copy_stream):
            e0.record(copy_stream)
            copy_fn()
            e1.record(copy_stream)
        e1.synchronize()
        ts_ms.append(e0.elapsed_time(e1))

    med = statistics.median(ts_ms)
    rel_std = statistics.pstdev(ts_ms) / med if med and trials > 1 else 0.0
    gbps = nbytes / (med / 1e3) / 1e9
    return gbps, rel_std


def start_sustained_load(op: Callable[[], None], holder_stream: torch.cuda.Stream) -> None:
    """Prime holder stream before a benchmark block."""
    for _ in range(4):
        topup_holder(op, holder_stream)


def drain_stream(stream: torch.cuda.Stream) -> None:
    e = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        e.record(stream)
    e.synchronize()

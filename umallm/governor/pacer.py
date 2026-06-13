"""PacedCopier -- one transfer as a rate-paced chunk train on its own stream.

The actuation primitive.  g9 showed CUDA stream priority cannot deprioritize a
copy and the direction null says the issuer does not matter, so the ONLY knob
that shapes a transfer's HBM footprint is its delivered rate, and the only way
to set the rate of a DMA that runs at link speed once issued is to chunk it
and space the launches:

    period T = chunk_bytes / rate ; gap = T - chunk_link_time

Enforcement is chunk-granular: every window of length T carries at most one
chunk, so the *windowed* rate never exceeds the lease (critiques F5/F6: epoch-
average pacing lets sub-epoch bursts through; this does not).  The rate is
re-read from `rate_fn` before every launch, so the slow trim loop applies to
the remainder of an in-flight transfer at chunk boundaries -- which is also
where cancellation lands.

Timing discipline: launches go to a dedicated stream on the SOURCE device,
completion is observed with stream.query() polling + a final stream-scoped
synchronize.  No whole-device synchronization anywhere (the g4 trap).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class CopyResult:
    nbytes: int
    n_chunks: int
    submit_t: float
    start_t: float = 0.0
    done_t: float = 0.0
    cancelled: bool = False
    chunks_launched: int = 0
    bytes_launched: int = 0
    paced_gap_s: float = 0.0          # total host-inserted gap time
    next_due_final: float = 0.0       # pacing clock at exit (train continuity)
    rate_samples: list = field(default_factory=list)  # (chunk_idx, rate_gbs)

    @property
    def duration_s(self) -> float:
        return max(self.done_t - self.start_t, 1e-9)

    @property
    def achieved_gbs(self) -> float:
        return self.nbytes / self.duration_s / 1e9


class PacedCopier:
    """depth=2 keeps one chunk in flight while the next is enqueued: without
    pipelining, a host poll between back-to-back 64 MB chunks costs ~27% of
    NVLink rate (measured 192.6 vs 264.7 GB/s on the A100 bridge), and the
    actuator could never deliver the rates it was commanded near link peak."""

    def __init__(self, device: int, poll_s: float = 100e-6, depth: int = 2):
        import torch
        self.torch = torch
        self.device = device
        self.poll_s = poll_s
        self.depth = max(1, depth)
        with torch.cuda.device(device):
            self.stream = torch.cuda.Stream(device=device)
            self._evs = [torch.cuda.Event() for _ in range(self.depth)]

    def run(self, src, dst, rate_fn, chunk_bytes: int,
            cancel: threading.Event | None = None,
            unpaced: bool = False,
            next_due0: float | None = None) -> CopyResult:
        """Copy flat tensor src -> dst (same numel/dtype) as a chunk train.

        rate_fn() -> GB/s, polled before each chunk; <=0 or inf means
        "no gap" for that chunk.  unpaced=True launches all chunks
        back-to-back (the NIXL-default analog) but still chunked, so the
        baseline and governed arms differ ONLY in pacing, not in kernel mix.
        next_due0 threads the pacing clock across consecutive run() calls so
        a sustained train (s1 census) has no burst at buffer-wrap boundaries.
        """
        torch = self.torch
        el_sz = src.element_size()
        n_el = src.numel()
        nbytes = n_el * el_sz
        chunk_el = max(1, chunk_bytes // el_sz)
        n_chunks = (n_el + chunk_el - 1) // chunk_el
        res = CopyResult(nbytes=nbytes, n_chunks=n_chunks,
                         submit_t=time.monotonic())

        with torch.cuda.device(self.device):
            res.start_t = time.monotonic()
            next_due = next_due0 if next_due0 is not None else res.start_t
            for ci in range(n_chunks):
                if cancel is not None and cancel.is_set():
                    res.cancelled = True
                    break
                if ci >= self.depth:
                    # bound the in-flight queue: wait for chunk ci-depth.
                    # blocks at most one chunk's DMA time; releases the GIL.
                    self._evs[ci % self.depth].synchronize()
                if not unpaced:
                    now = time.monotonic()
                    if now < next_due:
                        res.paced_gap_s += next_due - now
                        while time.monotonic() < next_due:
                            if cancel is not None and cancel.is_set():
                                res.cancelled = True
                                break
                            time.sleep(min(self.poll_s,
                                           max(0.0, next_due - time.monotonic())))
                        if res.cancelled:
                            break
                lo = ci * chunk_el
                hi = min(n_el, lo + chunk_el)
                with torch.cuda.stream(self.stream):
                    dst[lo:hi].copy_(src[lo:hi], non_blocking=True)
                    self._evs[ci % self.depth].record(self.stream)
                res.chunks_launched += 1
                res.bytes_launched += (hi - lo) * el_sz
                if not unpaced:
                    rate = rate_fn()
                    if ci < 32 or ci % 16 == 0:
                        res.rate_samples.append((ci, round(rate, 2)))
                    if rate and rate > 0 and rate != float("inf"):
                        # schedule from max(now, next_due): a late chunk does
                        # not earn back its delay as a burst credit
                        base = max(time.monotonic(), next_due)
                        next_due = base + (hi - lo) * el_sz / (rate * 1e9)
            self.stream.synchronize()      # stream-scoped: drain our chunks
            res.done_t = time.monotonic()
            res.next_due_final = next_due
        return res

"""Shared harness for the s* governor experiments.

Centralizes the two measurement-critical pieces so every s-experiment uses
the identical victim and the identical actuator:

  make_decoder -- the memory-bound Llama-3-8B-GQA decode victim used across
                  the g-experiments, timed with CUDA events on its own stream
                  (never a whole-device sync: the g4 trap).
  Train        -- a sustained paced copy train built on PacedCopier (the
                  governor's deployed actuator), with the pacing clock
                  threaded across buffer wraps so there is no burst at wrap
                  boundaries.  Used by the s1 census and the s3 accuracy
                  probes; s2 exercises the full Governor instead.
"""
from __future__ import annotations

import threading
import time

GEOM = dict(D=4096, H=32, HKV=8, HD=128, DFF=14336)   # Llama-3-8B GQA


def make_decoder(dev: str, batch: int, ctx: int):
    """Returns timed_iter() -> ms for one decode step on `dev` (warmed up)."""
    import torch
    import torch.nn.functional as F
    dt = torch.float16
    D, H, HKV, HD, DFF = (GEOM[k] for k in ("D", "H", "HKV", "HD", "DFF"))
    with torch.cuda.device(dev):
        W = {k: torch.randn(*s, dtype=dt, device=dev) * 0.02 for k, s in {
            "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD),
            "o": (H * HD, D), "g": (D, DFF), "u": (D, DFF),
            "d": (DFF, D)}.items()}
        Kc = torch.randn(batch, HKV, ctx, HD, dtype=dt, device=dev) * 0.02
        Vc = torch.randn(batch, HKV, ctx, HD, dtype=dt, device=dev) * 0.02
        x = torch.randn(batch, 1, D, dtype=dt, device=dev) * 0.02
        stream = torch.cuda.Stream(device=dev)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)

    def step():
        q = (x @ W["q"]).view(batch, 1, H, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(batch, 1, H * HD) @ W["o"]) + (
            F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    def timed_iter() -> float:
        with torch.cuda.device(dev), torch.cuda.stream(stream):
            e0.record(stream)
            step()
            e1.record(stream)
        e1.synchronize()
        return e0.elapsed_time(e1)

    for _ in range(30):
        with torch.cuda.device(dev), torch.cuda.stream(stream):
            step()
    stream.synchronize()
    return timed_iter


class Train:
    """Sustained paced copy train on a dedicated thread (governor actuator)."""

    def __init__(self, copier, src, dst, chunk_bytes: int):
        self.copier, self.src, self.dst = copier, src, dst
        self.chunk_bytes = chunk_bytes
        self.cancel = threading.Event()
        self.go = threading.Event()
        self._stop_train = threading.Event()   # replaced per start(); exists
        self._done = threading.Event()         # from birth (stop()-race fix)
        self.rate_gbs = 0.0
        self.bytes_launched = 0
        self.runs = 0
        self.t_start = 0.0
        self.t_stop = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            self.go.wait()
            if self.cancel.is_set():
                return
            self.bytes_launched = 0
            self.runs = 0
            self.t_start = time.monotonic()
            stop = threading.Event()
            self._stop_train = stop
            next_due = None
            while not stop.is_set():
                r = self.copier.run(self.src, self.dst,
                                    rate_fn=lambda: self.rate_gbs,
                                    chunk_bytes=self.chunk_bytes,
                                    cancel=stop, next_due0=next_due)
                next_due = r.next_due_final
                self.bytes_launched += r.bytes_launched
                self.runs += 1
            self.t_stop = time.monotonic()
            if self.cancel.is_set():
                self._done.set()
                return
            self.go.clear()
            self._done.set()

    def start(self, rate_gbs: float):
        self.rate_gbs = rate_gbs
        self._done = threading.Event()
        self.go.set()

    def stop(self) -> float:
        """Returns achieved GB/s over the train's life (launched bytes)."""
        self._stop_train.set()
        self._done.wait(timeout=30)
        dur = max(self.t_stop - self.t_start, 1e-9)
        return self.bytes_launched / dur / 1e9

    def shutdown(self):
        self.cancel.set()
        if hasattr(self, "_stop_train"):
            self._stop_train.set()
        self.go.set()
        self.copier.stream.synchronize()

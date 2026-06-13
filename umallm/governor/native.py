"""NativePacedCopier -- drop-in replacement for pacer.PacedCopier, off the GIL.

Wraps csrc/native_pacer.cpp (JIT-built via torch.utils.cpp_extension on first
use; cached under ~/.cache/torch_extensions).  Interface-compatible with
PacedCopier.run(src, dst, rate_fn, chunk_bytes, cancel, unpaced, next_due0)
so Train / Governor switch actuators without code changes:

  - the chunk train runs entirely in C++ with the GIL released (the Python
    pacer held the GIL for ~800-3000 launches/s next to the victim's timed
    loop -- the review's matched-cadence confound);
  - rate_fn/cancel semantics are preserved by a 20 Hz updater thread that
    mirrors them into the shared ControlBox (atomic rate + cancel flag) the
    C++ loop reads lock-free per chunk.  20 wakeups/s replaces thousands of
    GIL-holding launches/s; cancel latency is <=50ms + one chunk.

Falls back loudly: importing this module raises if the toolchain is missing;
callers use `available()` / Governor(pacer="auto") for graceful selection.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .pacer import CopyResult

_CSRC = Path(__file__).resolve().parent / "csrc" / "native_pacer.cpp"
_mod = None
_mod_err: Exception | None = None


def _load():
    global _mod, _mod_err
    if _mod is not None:
        return _mod
    if _mod_err is not None:
        raise _mod_err
    try:
        from torch.utils import cpp_extension
        _mod = cpp_extension.load(
            name="peerkv_native_pacer",
            sources=[str(_CSRC)],
            extra_cflags=["-O3", "-std=c++17"],
            with_cuda=True,            # include dirs + -lcudart (no .cu file)
            verbose=False,
        )
        return _mod
    except Exception as e:             # missing nvcc/toolchain etc.
        _mod_err = e
        raise


def available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


class NativePacedCopier:
    is_native = True

    def __init__(self, device: int, poll_s: float = 0.05, depth: int = 2):
        mod = _load()
        self.device = device
        self.depth = depth
        self.poll_s = poll_s            # rate_fn/cancel mirror cadence
        self._pacer = mod.NativePacer(device, depth)
        self._mod = mod
        # PacedCopier exposes .stream for the harness's stream-scoped drains;
        # the native pacer drains internally, so provide a no-op shim.
        self.stream = _NullStream()

    def run(self, src, dst, rate_fn, chunk_bytes: int,
            cancel: threading.Event | None = None,
            unpaced: bool = False,
            next_due0: float | None = None) -> CopyResult:
        el_sz = src.element_size()
        nbytes = src.numel() * el_sz
        box = self._mod.ControlBox()
        if not unpaced:
            try:
                r0 = rate_fn()
            except Exception:
                r0 = 0.0
            box.set_rate(float(r0) if r0 and r0 > 0 and r0 != float("inf")
                         else 0.0)
        if cancel is not None and cancel.is_set():
            box.cancel()

        stop_upd = threading.Event()

        def _mirror():
            while not stop_upd.wait(self.poll_s):
                if cancel is not None and cancel.is_set():
                    box.cancel()
                    return
                if not unpaced:
                    try:
                        r = rate_fn()
                    except Exception:
                        return
                    box.set_rate(float(r) if r and r > 0
                                 and r != float("inf") else 0.0)

        upd = None
        if cancel is not None or not unpaced:
            upd = threading.Thread(target=_mirror, daemon=True,
                                   name=f"native-pacer-mirror-d{self.device}")
            upd.start()
        submit_t = time.monotonic()
        try:
            d = self._pacer.run(dst.data_ptr(), src.data_ptr(), nbytes,
                                chunk_bytes, box, unpaced,
                                -1.0 if next_due0 is None else next_due0)
        finally:
            stop_upd.set()
            if upd is not None:
                upd.join(timeout=1.0)

        n_chunks = (nbytes + chunk_bytes - 1) // chunk_bytes
        res = CopyResult(nbytes=nbytes, n_chunks=n_chunks, submit_t=submit_t)
        res.start_t = d["start_t"]
        res.done_t = d["done_t"]
        res.cancelled = d["cancelled"]
        res.chunks_launched = d["chunks_launched"]
        res.bytes_launched = d["bytes_launched"]
        res.paced_gap_s = d["paced_gap_s"]
        res.next_due_final = d["next_due_final"]
        return res


class _NullStream:
    def synchronize(self) -> None:     # native run() drains before returning
        return

    def query(self) -> bool:
        return True

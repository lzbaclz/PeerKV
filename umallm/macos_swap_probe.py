"""macOS swap probe — addresses 100-round R16 (CUHK).

macOS doesn't expose per-process swap stats by default; we use `vm_stat`
output to extract pages_paged_in / pages_paged_out as a proxy for the
T3-tier round-trip volume. The probe samples vm_stat at a configurable
interval and emits deltas.

In the sandbox (Linux) we read /proc/meminfo's swap fields.
"""
from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, field


@dataclass
class SwapSnapshot:
    pages_in: int = 0
    pages_out: int = 0
    timestamp: float = 0.0


@dataclass
class SwapProbe:
    """Polls swap stats periodically; tracks deltas between samples."""
    last: SwapSnapshot = field(default_factory=SwapSnapshot)

    def sample(self) -> SwapSnapshot:
        if platform.system() == "Darwin":
            return self._sample_mac()
        return self._sample_linux()

    @staticmethod
    def _sample_mac() -> SwapSnapshot:
        try:
            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=2.0
            ).stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return SwapSnapshot(timestamp=time.time())
        pages_in = pages_out = 0
        for line in out.splitlines():
            if "Swapins" in line:
                pages_in = int(line.split(":")[1].strip().rstrip("."))
            elif "Swapouts" in line:
                pages_out = int(line.split(":")[1].strip().rstrip("."))
        return SwapSnapshot(pages_in=pages_in, pages_out=pages_out,
                            timestamp=time.time())

    @staticmethod
    def _sample_linux() -> SwapSnapshot:
        try:
            with open("/proc/vmstat") as fh:
                lines = fh.readlines()
        except FileNotFoundError:
            return SwapSnapshot(timestamp=time.time())
        pages_in = pages_out = 0
        for l in lines:
            if l.startswith("pswpin "):
                pages_in = int(l.split()[1])
            elif l.startswith("pswpout "):
                pages_out = int(l.split()[1])
        return SwapSnapshot(pages_in=pages_in, pages_out=pages_out,
                            timestamp=time.time())

    def delta(self) -> tuple[int, int]:
        cur = self.sample()
        dpin = max(0, cur.pages_in - self.last.pages_in)
        dpout = max(0, cur.pages_out - self.last.pages_out)
        self.last = cur
        return dpin, dpout

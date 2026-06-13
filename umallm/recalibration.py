"""Periodic re-calibration — addresses round-8 M1.

The cost-model scalars drift over time:
  - L2 miss latency rises under thermal throttling.
  - SoC bandwidth drops when other apps contend.
  - NVMe latency varies with disk wear.

We poll the probes every N minutes and re-write the calibration JSON.
Policy code reads from the cache on each step, so updates take effect
within one decode step.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .calibration import write_calibration


@dataclass
class RecalibrationDaemon:
    """Background thread that re-runs calibration every interval_s."""
    out_path: str
    interval_s: float = 300.0   # 5 minutes
    on_update: Callable | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread = field(default=None)

    def start(self) -> None:
        def run():
            while not self._stop.is_set():
                try:
                    cal = write_calibration(self.out_path)
                    if self.on_update is not None:
                        self.on_update(cal)
                except Exception:
                    pass
                if self._stop.wait(self.interval_s):
                    break
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

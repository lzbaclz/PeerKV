"""HeadroomEstimator -- NVML poller for ambient (un-admitted) traffic.

The ledger knows the traffic it admitted; this estimator sees the rest
(co-tenants, un-governed engines).  It polls, per GPU:

  - NVLink TX/RX cumulative throughput counters (FieldValues 138/139, KiB) --
    these work WITHOUT root, which is what makes the design portable to the
    no-sudo HGX container where DCGM DRAM_ACTIVE is unavailable;
  - utilization.gpu / utilization.memory as a coarse activity proxy.

ambient(dev, role) = max(0, measured link rate - rate the ledger admitted on
that endpoint).  Degrades to a zero estimator when pynvml is missing (CPU CI)
-- admission then trusts the ledger alone, which is exact on an idle box.
"""
from __future__ import annotations

import threading
import time


class HeadroomEstimator:
    def __init__(self, devices: list[int], poll_s: float = 0.05):
        self.devices = devices
        self.poll_s = poll_s
        self._lock = threading.Lock()
        self._snap: dict[int, dict] = {d: {"nvlink_tx_gbs": 0.0,
                                           "nvlink_rx_gbs": 0.0,
                                           "util_gpu": 0.0,
                                           "util_mem": 0.0} for d in devices}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_ok = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handles = {d: pynvml.nvmlDeviceGetHandleByIndex(d)
                             for d in devices}
            self._nvml_ok = True
        except Exception:                       # no GPU / no pynvml: zeros
            self._pynvml = None

    # -------------------------------------------------------------- polling
    def _read_counters(self, dev: int) -> tuple[float, float]:
        nv = self._pynvml
        fv = nv.nvmlDeviceGetFieldValues(self._handles[dev], [
            nv.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
            nv.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX])
        out = []
        for f in fv:
            out.append(float(f.value.ullVal) * 1024.0 if f.nvmlReturn == 0
                       else 0.0)               # bytes cumulative
        return out[0], out[1]

    def _loop(self) -> None:
        nv = self._pynvml
        prev = {d: self._read_counters(d) for d in self.devices}
        prev_t = time.monotonic()
        while not self._stop.wait(self.poll_s):
            now = time.monotonic()
            dt = max(now - prev_t, 1e-6)
            for d in self.devices:
                try:
                    tx, rx = self._read_counters(d)
                    util = nv.nvmlDeviceGetUtilizationRates(self._handles[d])
                    with self._lock:
                        self._snap[d] = {
                            "nvlink_tx_gbs": max(0.0, (tx - prev[d][0]) / dt / 1e9),
                            "nvlink_rx_gbs": max(0.0, (rx - prev[d][1]) / dt / 1e9),
                            "util_gpu": float(util.gpu),
                            "util_mem": float(util.memory),
                        }
                    prev[d] = (tx, rx)
                except Exception:
                    pass                        # transient NVML hiccup: keep last
            prev_t = now

    def start(self) -> "HeadroomEstimator":
        if self._nvml_ok and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="governor-estimator")
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------ interface
    def snapshot(self) -> dict[int, dict]:
        with self._lock:
            return {d: dict(v) for d, v in self._snap.items()}

    def ambient(self, ledger_rates: dict[int, dict[str, float]]) -> dict:
        """{dev: {'holder': gbs, 'receiver': gbs}} of un-admitted traffic.

        ledger_rates: what the ledger thinks it is driving on each endpoint
        (holder = egress = TX, receiver = ingress = RX).
        """
        snap = self.snapshot()
        out: dict[int, dict[str, float]] = {}
        for d, s in snap.items():
            mine = ledger_rates.get(d, {})
            out[d] = {
                "holder": max(0.0, s["nvlink_tx_gbs"] - mine.get("holder", 0.0)),
                "receiver": max(0.0, s["nvlink_rx_gbs"] - mine.get("receiver", 0.0)),
            }
        return out

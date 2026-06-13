"""CUDA box calibration (Track C M3 step 1; spec docs/MASTER_PLAN.md SS3.3).

``calibrate_box()`` measures, on the live box, the constants the cost model
depends on, and writes a per-box profile JSON:

    beta_hbm_gbps    streaming HBM read bandwidth      (vs e15's 773)
    beta_nvlink_gbps one-way peer-to-peer copy          (vs e15's 273)
    beta_pcie_gbps   pinned host->device copy           (vs e15's 24)
    c_nvlink_us      per-transfer setup, peer DMA       (vs e20's 23.6)
    c_pcie_us        per-transfer setup, pinned H2D     (vs e20's 12.6)
    c_hbm_us         per-launch setup, local kernel     (vs e20's 12.3)

``check_drift()`` compares a profile against the committed elastic_policy
constants and reports the relative drift -- the SS3.3 step-3 online loop
auto-degrades to SINGLE/TP when drift exceeds ``DRIFT_DEGRADE_THRESHOLD``.

NOTE: this is the *CUDA mainline* calibration. The legacy ``umallm.calibration``
(probe_soc_bandwidth / probe_l2_miss_latency) is a PARKED Track-D Apple-Silicon
probe and is unrelated (risk 9.A.5).

CLI:  python -m umallm.runtime.calibration [--out PATH] [--check]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .. import elastic_policy as _ep

STATUS = "IMPLEMENTED: startup box probe (SS3.3 step 1) + drift check (step 3 input)"

DRIFT_DEGRADE_THRESHOLD = 0.25     # fractional drift that trips auto-degrade
_MB = 1024 ** 2


@dataclass
class CalibrationResult:
    device: str
    n_gpu: int
    beta_hbm_gbps: float        # paged-KV gather (the e15 quantity)
    beta_hbm_stream_gbps: float  # contiguous streaming read (sanity vs peak)
    beta_nvlink_gbps: float
    beta_pcie_gbps: float
    c_nvlink_us: float
    c_pcie_us: float
    c_hbm_us: float
    measured_at: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _median_event_ms(fn, dev: "str", warmup: int = 3, reps: int = 11) -> float:
    """Median CUDA-event time of fn() on device's current stream."""
    import torch
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    ts = []
    for _ in range(reps):
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def calibrate_box(payload_mb: int = 512, hbm_gib: int = 1,
                  dev0: str = "cuda:0", dev1: str = "cuda:1") -> CalibrationResult:
    """Measure the cost-model constants on this box (~10 s, both GPUs idle).

    Bandwidths use a large payload (default 512 MB / 1 GiB); setup latencies
    use a 4 KiB payload where transfer time is negligible vs setup."""
    import torch
    assert torch.cuda.is_available(), "calibrate_box needs CUDA"
    n_gpu = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)

    torch.cuda.set_device(dev0)
    # --- HBM streaming read (sanity vs peak): fp16 sum over hbm_gib GiB ---
    n = hbm_gib * 1024 ** 3 // 2
    x = torch.empty(n, dtype=torch.float16, device=dev0).normal_()
    ms = _median_event_ms(lambda: x.sum(), dev0)
    beta_stream = (n * 2) / (ms * 1e-3) / 1e9
    tiny = torch.empty(2048, dtype=torch.float16, device=dev0)  # 4 KiB
    c_hbm_us = _median_event_ms(lambda: tiny.sum(), dev0) * 1e3

    # --- HBM effective KV read (the e15 quantity): time the real decode
    # attention kernel over a resident 32K KV and divide the KV bytes it must
    # read by the kernel time. This is exactly what the cost model's kv_read
    # term sees -- not a synthetic streaming number.
    import torch.nn.functional as F
    Hq, Hkv, D, S = 32, 8, 128, 32768
    q = torch.empty(1, Hq, 1, D, dtype=torch.float16, device=dev0).normal_()
    K = torch.empty(1, Hkv, S, D, dtype=torch.float16, device=dev0).normal_()
    V = torch.empty_like(K).normal_()
    try:
        fn = lambda: F.scaled_dot_product_attention(q, K, V, enable_gqa=True)
        fn()
    except TypeError:                       # older torch: expand KV heads
        Ke = K.repeat_interleave(Hq // Hkv, dim=1)
        Ve = V.repeat_interleave(Hq // Hkv, dim=1)
        fn = lambda: F.scaled_dot_product_attention(q, Ke, Ve)
    ms = _median_event_ms(fn, dev0)
    kv_bytes = 2 * Hkv * S * D * 2          # K+V, fp16
    beta_kv = kv_bytes / (ms * 1e-3) / 1e9
    del q, K, V

    # --- PCIe: pinned H2D ---
    nb = payload_mb * _MB
    host = torch.empty(nb // 2, dtype=torch.float16, pin_memory=True)
    dst0 = torch.empty_like(host, device=dev0)
    ms = _median_event_ms(lambda: dst0.copy_(host, non_blocking=True), dev0)
    beta_pcie = nb / (ms * 1e-3) / 1e9
    host4k = torch.empty(2048, dtype=torch.float16, pin_memory=True)
    dst4k = torch.empty_like(host4k, device=dev0)
    c_pcie_us = _median_event_ms(lambda: dst4k.copy_(host4k, non_blocking=True), dev0) * 1e3

    # --- NVLink one-way peer copy (if a second GPU exists) ---
    if n_gpu >= 2:
        src1 = torch.empty(nb // 2, dtype=torch.float16, device=dev1).normal_()
        dstp = torch.empty_like(src1, device=dev0)
        ms = _median_event_ms(lambda: dstp.copy_(src1, non_blocking=True), dev0)
        beta_nvl = nb / (ms * 1e-3) / 1e9
        s4k = torch.empty(2048, dtype=torch.float16, device=dev1)
        d4k = torch.empty_like(s4k, device=dev0)
        c_nvl_us = _median_event_ms(lambda: d4k.copy_(s4k, non_blocking=True), dev0) * 1e3
    else:
        beta_nvl, c_nvl_us = 0.0, float("inf")

    return CalibrationResult(
        device=name, n_gpu=n_gpu,
        beta_hbm_gbps=round(beta_kv, 1),
        beta_hbm_stream_gbps=round(beta_stream, 1),
        beta_nvlink_gbps=round(beta_nvl, 1),
        beta_pcie_gbps=round(beta_pcie, 1),
        c_nvlink_us=round(c_nvl_us, 1),
        c_pcie_us=round(c_pcie_us, 1),
        c_hbm_us=round(c_hbm_us, 1),
        measured_at=datetime.now(timezone.utc).isoformat())


def check_drift(result: CalibrationResult) -> dict:
    """Fractional drift of each live constant vs the committed e15/e20 values.

    The degrade gate compares LINK bandwidths only (NVLink, PCIe): those are
    measured the same way as the committed values and are what actually moves
    operationally (MIG re-enabled, links down -> the R1-route axis). The
    HBM-effective number is kernel-dependent (e15's 773 came from an older
    kernel; the working model re-fits beta_eff from decode anchors anyway), so
    it is reported as informational, never a degrade trigger."""
    ref = {
        "beta_hbm_gbps": _ep.BETA_HBM_GBPS,
        "beta_nvlink_gbps": _ep.BETA_NVLINK_GBPS,
        "beta_pcie_gbps": _ep.BETA_PCIE_GBPS,
        "c_nvlink_us": _ep.C_NVLINK_US,
        "c_pcie_us": _ep.C_PCIE_US,
        "c_hbm_us": _ep.C_HBM_US,
    }
    live = result.to_json()
    drift = {k: round((live[k] - v) / v, 4) for k, v in ref.items() if v}
    link_drift = [abs(drift[k]) for k in
                  ("beta_nvlink_gbps", "beta_pcie_gbps")]
    return {"drift": drift,
            "hbm_note": ("beta_hbm is kernel-effective and informational; "
                         "model beta_eff is re-fit from decode anchors"),
            "max_link_drift": max(link_drift),
            "degrade": max(link_drift) > DRIFT_DEGRADE_THRESHOLD}


def default_profile_path() -> Path:
    import torch
    tag = torch.cuda.get_device_name(0).replace(" ", "_") if torch.cuda.is_available() else "cpu"
    return Path.home() / ".peerkv" / f"calib_{tag}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true",
                    help="report drift vs committed e15/e20 constants")
    ap.add_argument("--payload-mb", type=int, default=512)
    args = ap.parse_args()

    t0 = time.time()
    res = calibrate_box(payload_mb=args.payload_mb)
    out = args.out or default_profile_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_module": "umallm.runtime.calibration", **res.to_json()}
    if args.check:
        payload["check"] = check_drift(res)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"-> {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

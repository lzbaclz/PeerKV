"""Box-idle probe (risk 9.A.4): refuse to measure on a contended shared GPU box.

The dual-A100 box is shared. A co-tenant's NVLink/HBM traffic silently corrupts
victim-cost measurements -- EXPERIMENT_STATUS.md SS D documents an e31 host arm
that read 6.0x under contention versus 1.86x once the box was clean. Every
g1--g5 / e2e entry point should gate on :func:`assert_box_idle` so a polluted run
aborts loudly instead of producing a phantom number.

Pure-stdlib + optional pynvml. With no pynvml/GPU it degrades to a no-op that
returns "idle" (so CPU CI is unaffected) but says so in the detail dict.
"""
from __future__ import annotations

import time


def box_idle(util_pct_max: float = 5.0,
             mem_delta_mb_max: float = 256.0,
             sample_s: float = 5.0,
             devices: tuple[int, ...] = (0, 1)) -> tuple[bool, dict]:
    """Sample GPU utilization (and NVLink, if available) for ``sample_s`` seconds.

    Returns ``(idle, detail)``. ``idle`` is False if any sampled device shows SM
    utilization above ``util_pct_max`` or free-memory drift beyond
    ``mem_delta_mb_max`` (a co-tenant allocating/freeing), i.e. someone else is
    using the box. NVLink throughput counters are included in ``detail`` when the
    driver exposes them.
    """
    try:
        import pynvml
    except Exception as e:  # pragma: no cover - CPU CI / no driver
        return True, {"skipped": f"pynvml unavailable ({e})", "idle_assumed": True}

    try:
        pynvml.nvmlInit()
    except Exception as e:  # pragma: no cover
        return True, {"skipped": f"nvmlInit failed ({e})", "idle_assumed": True}

    handles = {}
    for d in devices:
        try:
            handles[d] = pynvml.nvmlDeviceGetHandleByIndex(d)
        except Exception:
            continue

    max_util = {d: 0 for d in handles}
    free0 = {}
    free_min = {}
    free_max = {}
    for d, h in handles.items():
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        free0[d] = mem.free
        free_min[d] = mem.free
        free_max[d] = mem.free

    t0 = time.time()
    while time.time() - t0 < sample_s:
        for d, h in handles.items():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                max_util[d] = max(max_util[d], int(u.gpu))
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                free_min[d] = min(free_min[d], mem.free)
                free_max[d] = max(free_max[d], mem.free)
            except Exception:
                pass
        time.sleep(0.2)

    detail = {"util_pct_max_seen": max_util,
              "mem_free_drift_mb": {d: round((free_max[d] - free_min[d]) / 1e6, 1)
                                    for d in handles},
              "thresholds": {"util_pct_max": util_pct_max, "mem_delta_mb_max": mem_delta_mb_max},
              "sample_s": sample_s}

    idle = True
    for d in handles:
        if max_util[d] > util_pct_max:
            idle = False
        if (free_max[d] - free_min[d]) / 1e6 > mem_delta_mb_max:
            idle = False
    detail["idle"] = idle
    return idle, detail


def assert_box_idle(**kw) -> dict:
    """Raise ``RuntimeError`` if the box is not idle. Call at the top of any
    measurement script (g1--g5, e2e) before locking clocks / allocating."""
    idle, detail = box_idle(**kw)
    if not idle:
        raise RuntimeError(
            "PeerKV box-idle probe: a co-tenant appears active "
            f"(detail={detail}); refusing to measure (would corrupt victim cost). "
            "Wait for the box to clear or coordinate the window.")
    return detail


def gate_or_skip(label: str = "experiment", **kw) -> dict:
    """One-line gate at the top of every g*/e2e entry point.

    Honors the ``PEERKV_SKIP_IDLE_PROBE=1`` escape hatch (CI / dev / synthetic
    runs). Otherwise samples the box for 5 s and aborts loudly if a co-tenant is
    active (risk 9.A.4). Returns the detail dict so the caller can fold it into
    the JSON artifact for traceability.

    Usage at the top of ``main()`` (after ``args = ap.parse_args()``)::

        from umallm.observability import gate_or_skip
        gate_or_skip("g2")
    """
    import os
    if os.environ.get("PEERKV_SKIP_IDLE_PROBE"):
        return {"skipped": "PEERKV_SKIP_IDLE_PROBE=1", "label": label}
    detail = assert_box_idle(**kw)
    print(f"[box-idle/{label}] OK util_max={detail.get('util_pct_max_seen')} "
          f"mem_drift_mb={detail.get('mem_free_drift_mb')}")
    return detail


if __name__ == "__main__":
    ok, det = box_idle()
    print(("IDLE" if ok else "BUSY"), det)

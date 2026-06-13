"""PeerKV Prometheus metric set (spec: 04_cross_cutting.md SS2.2).

STATUS: defined + import-safe, but NOT YET WIRED into the connector/selector hot
path (risk 9.C.1: this closes the "metric names exist" half; the call-site
instrumentation in peerkv_attn.py / select_point is the remaining M2-M3 work).
If ``prometheus_client`` is absent the names degrade to no-op shims so importing
this module never fails in CPU CI.
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Gauge, Histogram
    _HAVE_PROM = True
except Exception:  # pragma: no cover - prometheus_client optional
    _HAVE_PROM = False

    class _Noop:
        def __init__(self, *a, **k): pass
        def labels(self, *a, **k): return self
        def inc(self, *a, **k): pass
        def set(self, *a, **k): pass
        def observe(self, *a, **k): pass

    Counter = Gauge = Histogram = _Noop  # type: ignore

STATUS = ("WIRED: selector metrics emitted by runtime.selector.online_select; "
          "violation counter also on the peerkv_attn hot path (_guard_route). "
          "TPOT/TTFT/XFER call-sites land with the formal connector (M2).")

# --- selector / do-no-harm ---
SELECTED_POINT = Counter("peerkv_selected_point_total", "corner chosen", ["point"])
DO_NO_HARM_VIOL = Counter("peerkv_do_no_harm_violations_total", "invariant breaches", ["rule"])
ADMISSIBLE_POINTS = Gauge("peerkv_admissible_points", "size of admissible set (last req)")

# --- latency (mode = single|peerkv, point=...) ---
TPOT = Histogram("peerkv_tpot_seconds", "time per output token", ["mode", "point"],
                 buckets=(.005, .01, .02, .04, .08, .16, .32, .64, 1.28))
TTFT = Histogram("peerkv_ttft_seconds", "time to first token", ["mode"],
                 buckets=(.05, .1, .2, .4, .8, 1.6, 3.2, 6.4))

# --- peer / link health ---
PEER_BUSY = Gauge("peerkv_peer_compute_busy", "1 if peer GPU is computing", ["peer"])
LINK_EFF_GBPS = Gauge("peerkv_link_eff_gbps", "measured one-way eff bw", ["fabric"])

# --- transfer (dir = push|pull; bytes is the footprint that drives victim cost) ---
XFER_BYTES = Counter("peerkv_xfer_bytes_total", "cross-GPU KV bytes", ["dir"])

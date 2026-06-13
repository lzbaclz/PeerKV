"""Governor -- quote / submit / feed_victim / report.

The seam an engine talks to.  An engine (or experiment) submits transfers;
the governor prices each one on BOTH endpoints with the calibrated curves,
admits it at the largest rate that keeps every endpoint under its eps budget
(additive across concurrent leases), and executes it as a paced chunk train.
Transfers that cannot get min_rate are DEFERRED (FIFO per source device) and
re-tried every defer_poll_s -- queueing delay is the visible, bounded price of
protection, reported per transfer so over-protection is measurable (the
strong-static baseline hides it).

Modes:
  'ff' -- feedforward only: calibrated curves, worst-case workload bucket,
          trim pinned at 1.0.  Protection without any victim feed.
  'fb' -- two-timescale: + per-device BudgetTrim driven by engine-reported
          victim iteration times (feed_victim).  Opens the throttle when the
          victim measures insensitive, backs off when the curve under-prices.

Thread model: one worker thread + PacedCopier per source device (transfers on
the same link serialize, matching one-DMA-engine reality; different sources
run concurrently -- the additive ledger is what keeps their sum safe).  The
epoch thread drives trims.  Victim devices never issue or poll anything.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field

from .budget import Ledger
from .calib import GovernorCalibration
from .control import BudgetTrim
from .estimator import HeadroomEstimator
from .pacer import PacedCopier, CopyResult


@dataclass
class TransferRequest:
    src: object                  # flat torch tensor (source buffer)
    dst: object                  # flat torch tensor (destination buffer)
    src_dev: int | None          # None = host
    dst_dev: int | None
    route: str = "peer"
    meta: dict = field(default_factory=dict)


@dataclass
class TransferHandle:
    req: TransferRequest
    submit_t: float
    done: threading.Event = field(default_factory=threading.Event)
    admit_t: float | None = None
    result: CopyResult | None = None
    lease_rate_gbs: float | None = None
    deferrals: int = 0
    error: BaseException | None = None    # worker survives; caller inspects

    def wait(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)

    @property
    def queue_delay_s(self) -> float:
        return (self.admit_t or self.submit_t) - self.submit_t

    @property
    def total_latency_s(self) -> float | None:
        return None if self.result is None else self.result.done_t - self.submit_t


class Governor:
    def __init__(self, calib: GovernorCalibration,
                 eps_holder_pct: float = 5.0, eps_receiver_pct: float = 5.0,
                 mode: str = "fb", chunk_mb: int = 64,
                 epoch_s: float = 0.25, defer_poll_s: float = 0.002,
                 min_rate_gbs: float = 4.0,
                 estimator: HeadroomEstimator | None = None,
                 pacer: str = "auto"):
        assert mode in ("ff", "fb")
        assert pacer in ("auto", "python", "native")
        self.calib = calib
        self.mode = mode
        # actuator selection: the native (C++/GIL-free) pacer tracks
        # commanded rates to ~1-2% across the full link range and removes
        # the host-launch confound; the Python pacer remains for CPU-only
        # environments and A/B experiments (s7)
        self._pacer_cls = PacedCopier
        self.pacer_kind = "python"
        if pacer in ("auto", "native"):
            try:
                from .native import NativePacedCopier, available
                if available():
                    self._pacer_cls = NativePacedCopier
                    self.pacer_kind = "native"
                elif pacer == "native":
                    raise RuntimeError("native pacer requested but toolchain "
                                       "unavailable")
            except ImportError:
                if pacer == "native":
                    raise
        self.chunk_bytes = chunk_mb << 20
        self.epoch_s = epoch_s
        self.defer_poll_s = defer_poll_s
        self.ledger = Ledger(calib, eps_holder_pct, eps_receiver_pct,
                             min_rate_gbs)
        self.estimator = estimator
        if estimator is not None:
            estimator.start()              # Governor owns the lifecycle
        self.eps = {"holder": eps_holder_pct, "receiver": eps_receiver_pct}
        # Trim clamps coupled to the budget (review critical #1): the floor
        # keeps m*invert(eps) >= min_rate so deep backoff cannot enter the
        # absorbing all-deferred state; the cap keeps m*invert(eps) inside
        # the calibrated census range, beyond which the curve says nothing.
        eps_t = min(eps_holder_pct, eps_receiver_pct)
        try:
            w = calib.curve("receiver", "peer", "worst")
            inv = max(w.invert(eps_t), 1e-6)
            self._m_floor = max(0.05, min_rate_gbs / inv)
            self._m_cap = max(1.0, max(p.rate_gbs for p in w.points) / inv)
        except KeyError:
            self._m_floor, self._m_cap = 0.05, 16.0
        self.trims: dict[int, BudgetTrim] = {}
        self._victim_feed: dict[int, collections.deque] = {}
        self._empty_epochs: dict[int, int] = {}
        self._queues: dict[int, collections.deque] = {}
        self._pending_dst: collections.Counter = collections.Counter()
        self._next_due: dict[int, float] = {}    # pacing clock per source
        self._workers: dict[int, threading.Thread] = {}
        self._copiers: dict[int, PacedCopier] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stats = collections.Counter()
        self._last_activity = 0.0      # last admit/release, for idle epochs
        self._epoch_thread: threading.Thread | None = None
        if mode == "fb":
            self._epoch_thread = threading.Thread(
                target=self._epoch_loop, daemon=True, name="governor-epochs")
            self._epoch_thread.start()

    # -------------------------------------------------------------- victims
    def feed_victim(self, dev: int, iter_ms: float) -> None:
        """Engine-reported per-iteration decode time for GPU `dev`."""
        with self._lock:
            dq = self._victim_feed.setdefault(
                dev, collections.deque(maxlen=65536))
            self.trims.setdefault(dev, BudgetTrim(
                min(self.eps["holder"], self.eps["receiver"]),
                m_min=self._m_floor, m_max=self._m_cap))
        dq.append((time.monotonic(), iter_ms))

    def _trim_view(self) -> dict[int, float]:
        if self.mode != "fb":
            return {}
        return {d: t.multiplier for d, t in self.trims.items()}

    def _epoch_loop(self) -> None:
        while not self._stop.wait(self.epoch_s):
            cutoff = time.monotonic() - self.epoch_s
            active = self.ledger.active_leases()
            busy_devs = {l.src_dev for l in active} | {l.dst_dev for l in active}
            with self._lock:
                feeds = list(self._victim_feed.items())
                pending_dst = dict(self._pending_dst)
            quiet = (time.monotonic() - self._last_activity) >= self.epoch_s
            ambient_quiet = True
            if self.estimator is not None:
                snap = self.estimator.snapshot()
                ambient_quiet = all(
                    s.get("nvlink_tx_gbs", 0) + s.get("nvlink_rx_gbs", 0) < 2.0
                    for s in snap.values())
            for dev, dq in feeds:
                samples = [ms for (t, ms) in list(dq) if t >= cutoff]
                trim = self.trims.get(dev)
                if trim is None:
                    continue
                # work_pending is per-VICTIM (review major: a global flag let
                # no-traffic victims blind-probe to m_max)
                mine = pending_dst.get(dev, 0) > 0 or any(
                    l.dst_dev == dev or l.src_dev == dev for l in active)
                if dev in busy_devs or not quiet:
                    if not samples and len(dq):
                        # previously-alive feed went silent under our traffic.
                        # One empty epoch is usually telemetry burstiness
                        # (engine metrics update in clumps at request
                        # completions); SUSTAINED silence is the victim-
                        # collapse signature. Grace of 3 epochs, then treat
                        # missing feed as danger.
                        self._empty_epochs[dev] = \
                            self._empty_epochs.get(dev, 0) + 1
                        if self._empty_epochs[dev] >= 3:
                            trim.observe_starved()
                    else:
                        self._empty_epochs[dev] = 0
                        trim.observe_loaded(samples, work_pending=mine)
                elif ambient_quiet:
                    # baseline only when our ledger AND the NVML ambient view
                    # are quiet (review major: EMA poisoning by co-tenants)
                    trim.observe_idle(samples)

    # ------------------------------------------------------------ admission
    def _ambient(self) -> dict:
        if self.estimator is None:
            return {}
        rates: dict[int, dict[str, float]] = {}
        for l in self.ledger.active_leases():
            if l.src_dev is not None:
                rates.setdefault(l.src_dev, {}).setdefault("holder", 0.0)
                rates[l.src_dev]["holder"] += l.rate_gbs
            if l.dst_dev is not None:
                rates.setdefault(l.dst_dev, {}).setdefault("receiver", 0.0)
                rates[l.dst_dev]["receiver"] += l.rate_gbs
        return self.estimator.ambient(rates)

    def quote(self, src_dev: int | None, dst_dev: int | None,
              nbytes: int, route: str = "peer",
              workload_hint: dict | None = None) -> dict:
        """Price a prospective transfer without admitting it."""
        rate = self.ledger.max_admissible_rate(
            src_dev, dst_dev, route, self._ambient(), workload_hint,
            self._trim_view() or 1.0)
        hint = workload_hint or {}
        out = {"admissible_rate_gbs": round(rate, 2),
               "eta_s": round(nbytes / max(rate, 1e-3) / 1e9, 4),
               "deferred": rate < self.ledger.min_rate_gbs}
        for dev, role in ((src_dev, "holder"), (dst_dev, "receiver")):
            if dev is not None:
                out[f"predicted_cost_{role}_pct"] = round(
                    self.ledger.predicted_cost(dev, role, route, rate,
                                               hint.get(dev, "worst")), 3)
        return out

    # ------------------------------------------------------------ execution
    def submit(self, req: TransferRequest) -> TransferHandle:
        h = TransferHandle(req=req, submit_t=time.monotonic())
        key = req.src_dev if req.src_dev is not None else req.dst_dev
        with self._lock:
            self._queues.setdefault(key, collections.deque()).append(h)
            if req.dst_dev is not None:
                self._pending_dst[req.dst_dev] += 1
            if key not in self._workers:
                # pacer stream lives on the issuing device; for host sources
                # the destination GPU issues (direction is a null -- g3/g11)
                self._copiers[key] = self._pacer_cls(device=key)
                w = threading.Thread(target=self._worker_loop, args=(key,),
                                     daemon=True, name=f"governor-src{key}")
                self._workers[key] = w
                w.start()
            self._stats["submitted"] += 1
        return h

    def _worker_loop(self, key: int) -> None:
        copier = self._copiers[key]
        while not self._stop.is_set():
            with self._lock:
                q = self._queues.get(key)
                h = q.popleft() if q else None
            if h is None:
                time.sleep(self.defer_poll_s)
                continue
            try:
                self._serve_one(key, copier, h)
            except BaseException as e:           # noqa: BLE001 -- a CUDA
                h.error = e                      # error must not kill the
                with self._lock:                 # worker and starve the queue
                    self._stats["transfer_errors"] += 1
            finally:
                if h.req.dst_dev is not None:
                    with self._lock:
                        self._pending_dst[h.req.dst_dev] -= 1
                h.done.set()

    def _serve_one(self, key: int, copier: PacedCopier,
                   h: TransferHandle) -> None:
        req = h.req
        hint = req.meta.get("workload_hint")
        lease = None
        while lease is None:
            if self._stop.is_set():
                return
            lease = self.ledger.admit(
                req.src_dev, req.dst_dev, req.route,
                req.src.numel() * req.src.element_size(),
                self._ambient(), hint, self._trim_view() or 1.0)
            if lease is None:
                h.deferrals += 1
                with self._lock:
                    self._stats["deferral_polls"] += 1
                time.sleep(self.defer_poll_s)
        h.admit_t = time.monotonic()
        self._last_activity = h.admit_t
        h.lease_rate_gbs = lease.rate_gbs
        trim0 = (self._trim_view() or {}).get(req.dst_dev, 1.0) or 1.0
        link = self.calib.link_peak_gbs.get(req.route, float("inf"))
        ledger, base_rate = self.ledger, lease.rate_gbs

        def rate_fn():
            # base allocation from admission, trim ratio applied live so the
            # slow loop reshapes the in-flight tail at chunk grain; the lease
            # record is updated so accounting tracks actuation (review major)
            t_now = (self._trim_view() or {}).get(req.dst_dev, 1.0) or 1.0
            r = min(link, base_rate * (t_now / trim0))
            if abs(r - lease.rate_gbs) > 0.5:
                ledger.update_rate(lease, r)
            return r

        try:
            # thread the pacing clock across transfers on this source so
            # back-to-back small payloads cannot burst at chunk 0
            # (review major: intra-transfer-only pacing)
            h.result = copier.run(req.src, req.dst, rate_fn, self.chunk_bytes,
                                  cancel=self._stop,
                                  next_due0=self._next_due.get(key))
            self._next_due[key] = h.result.next_due_final
        finally:
            self._last_activity = time.monotonic()   # BEFORE release: the
            self.ledger.release(lease)               # epoch loop must not
            with self._lock:                         # see (no-lease, stale
                self._stats["completed"] += 1        # activity) and call it
                                                     # an idle epoch

    # ------------------------------------------------------------- shutdown
    def report(self) -> dict:
        return {
            "mode": self.mode,
            "pacer": self.pacer_kind,
            "stats": dict(self._stats),
            "active_leases": len(self.ledger.active_leases()),
            "trims": {d: t.snapshot() for d, t in self.trims.items()},
        }

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for w in self._workers.values():
            w.join(timeout=timeout)
        for c in self._copiers.values():    # no in-flight chunk bleeds into
            c.stream.synchronize()          # whatever runs after us
        if self._epoch_thread is not None:
            self._epoch_thread.join(timeout=timeout)

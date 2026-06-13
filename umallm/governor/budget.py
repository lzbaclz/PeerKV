"""Ledger -- dual-ended rate leases with additive admission in COST space.

Every in-flight transfer holds a Lease: a delivered-rate allocation charged to
BOTH endpoints -- the source's read-polarity (holder) account and the
destination's write-polarity (receiver) account.

Within one route, the g8 additive law says the victim feels the AGGREGATE
rate, so the aggregate goes through that route's curve.  ACROSS routes the
rates are not commensurable -- a host-route GB/s costs ~5-8x a peer GB/s on
the same victim port (s1: receiver/host/b1 +5.6% @24 GB/s vs receiver/peer/b1
+0.9% @31 GB/s) -- so admission sums COSTS, not rates (adversarial-review
critical #2):

    admissible extra rate x on route r at endpoint (dev, role):
        cost_other = sum_{r' != r} curve_{role,r'}.predict_upper(agg_{r'} + amb_{r'})
        x = curve_{role,r}.invert(eps - cost_other) - agg_r - amb_r

Cross-route cost additivity is a conservative assumption (both routes land on
the same HBM port).  Admission and lease insertion happen under ONE lock
(adversarial-review critical #3: the compute-then-insert race over-admitted
up to KxBudget for K racing sources).

Host-route endpoints: the CPU side of a host transfer has no GPU victim; only
the GPU endpoint is charged.  Thread-safe; pure Python.
"""
from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field

from .calib import GovernorCalibration

_ids = itertools.count(1)


@dataclass
class Lease:
    lease_id: int
    src_dev: int | None          # None = host
    dst_dev: int | None
    route: str                   # 'peer' | 'host'
    rate_gbs: float
    nbytes: int
    workload_hint: dict = field(default_factory=dict)  # dev -> bucket name


class Ledger:
    def __init__(self, calib: GovernorCalibration,
                 eps_holder_pct: float = 5.0,
                 eps_receiver_pct: float = 5.0,
                 min_rate_gbs: float = 4.0):
        self.calib = calib
        self.eps = {"holder": eps_holder_pct, "receiver": eps_receiver_pct}
        self.min_rate_gbs = min_rate_gbs
        self._lock = threading.Lock()
        self._active: dict[int, Lease] = {}

    # ----------------------------------------------------------- accounting
    def _routes_for(self, role: str) -> list[str]:
        seen = []
        for k in self.calib.curves:
            r, route, _ = k.split("/")
            if r == role and route not in seen:
                seen.append(route)
        return seen

    def _allocated_locked(self, dev: int, role: str, route: str) -> float:
        tot = 0.0
        for l in self._active.values():
            if l.route != route:
                continue
            if role == "holder" and l.src_dev == dev:
                tot += l.rate_gbs
            elif role == "receiver" and l.dst_dev == dev:
                tot += l.rate_gbs
        return tot

    def allocated(self, dev: int, role: str, route: str | None = None) -> float:
        with self._lock:
            if route is not None:
                return self._allocated_locked(dev, role, route)
            return sum(self._allocated_locked(dev, role, r)
                       for r in self._routes_for(role))

    def active_leases(self) -> list[Lease]:
        with self._lock:
            return list(self._active.values())

    # ------------------------------------------------------------ admission
    def _endpoint_cap_locked(self, dev: int | None, role: str, route: str,
                             ambient: dict, hint: str,
                             trim: "float | dict") -> float:
        if dev is None:                       # host endpoint: no GPU victim
            return float("inf")
        t = trim.get(dev, 1.0) if isinstance(trim, dict) else trim
        amb_dev = ambient.get(dev, {}) if ambient else {}
        # cost already charged by OTHER routes on this endpoint
        cost_other = 0.0
        for r2 in self._routes_for(role):
            if r2 == route:
                continue
            agg2 = self._allocated_locked(dev, role, r2)
            amb2 = amb_dev.get(f"{role}_{r2}", 0.0)
            if agg2 + amb2 > 0:
                cost_other += self.calib.curve(role, r2, hint).predict_upper(
                    agg2 + amb2)
        eps_left = max(0.0, self.eps[role] - cost_other)
        curve = self.calib.curve(role, route, hint)
        budget = curve.invert(eps_left) * t
        amb_r = amb_dev.get(role, 0.0) + amb_dev.get(f"{role}_{route}", 0.0)
        return budget - self._allocated_locked(dev, role, route) - amb_r

    def _max_admissible_locked(self, src_dev, dst_dev, route,
                               ambient, hint, trim) -> float:
        ambient = ambient or {}
        hint = hint or {}
        caps = [
            self._endpoint_cap_locked(src_dev, "holder", route, ambient,
                                      hint.get(src_dev, "worst"), trim),
            self._endpoint_cap_locked(dst_dev, "receiver", route, ambient,
                                      hint.get(dst_dev, "worst"), trim),
        ]
        link = self.calib.link_peak_gbs.get(route)
        if link:
            used = sum(l.rate_gbs for l in self._active.values()
                       if l.route == route and l.src_dev == src_dev
                       and l.dst_dev == dst_dev)
            caps.append(link - used)
        return max(0.0, min(caps))

    def max_admissible_rate(self, src_dev: int | None, dst_dev: int | None,
                            route: str,
                            ambient: dict | None = None,
                            workload_hint: dict | None = None,
                            trim: "float | dict" = 1.0) -> float:
        """Largest rate a new (src->dst, route) lease may take right now.

        ambient: {dev: {'holder': gbs, 'receiver': gbs, ...}} un-admitted
        traffic (optionally per-route via 'role_route' keys).
        """
        with self._lock:
            return self._max_admissible_locked(src_dev, dst_dev, route,
                                               ambient, workload_hint, trim)

    def admit(self, src_dev: int | None, dst_dev: int | None, route: str,
              nbytes: int,
              ambient: dict | None = None,
              workload_hint: dict | None = None,
              trim: "float | dict" = 1.0) -> Lease | None:
        """Grant a lease at the max admissible rate, or None (caller defers).

        Compute + insert are ATOMIC under one lock: racing sources cannot
        admit against the same residual budget.
        """
        with self._lock:
            rate = self._max_admissible_locked(src_dev, dst_dev, route,
                                               ambient, workload_hint, trim)
            if rate < self.min_rate_gbs:
                return None
            lease = Lease(next(_ids), src_dev, dst_dev, route, rate, nbytes,
                          workload_hint or {})
            self._active[lease.lease_id] = lease
            return lease

    def update_rate(self, lease: Lease, rate_gbs: float) -> None:
        """Keep accounting in step with live actuation (trim rescaling)."""
        with self._lock:
            if lease.lease_id in self._active:
                lease.rate_gbs = rate_gbs

    def release(self, lease: Lease) -> None:
        with self._lock:
            self._active.pop(lease.lease_id, None)

    # ------------------------------------------------------------- estimate
    def predicted_cost(self, dev: int, role: str, route: str,
                       extra_rate_gbs: float = 0.0, hint: str = "worst") -> float:
        """Central-estimate victim slowdown (%) summing cost across routes."""
        with self._lock:
            total = 0.0
            for r2 in self._routes_for(role):
                agg = self._allocated_locked(dev, role, r2)
                if r2 == route:
                    agg += extra_rate_gbs
                if agg > 0:
                    total += self.calib.curve(role, r2, hint).predict(agg)
            return total

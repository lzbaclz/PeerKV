"""Footprint->victim-cost curves from the s1 real-copy census, invertible.

Replaces the g10 paced-*injector* curves with curves measured from REAL paced
copy chunk trains (s1_governor_calib.py), because the injector mis-prices both
endpoints (g11: receiver measured +23.5% vs +11.8% predicted; g7: holder
sustained +1.1% vs ~4% predicted).  The curve key is

    (role, route, workload)

  role     'holder'   -- the copy READS this endpoint's HBM (source side)
           'receiver' -- the copy WRITES this endpoint's HBM (destination side)
  route    'peer' (NVLink) | 'host' (PCIe)
  workload sensitivity bucket of the victim decode, e.g. 'b1' (batch=1,
           maximally sensitive) / 'b8'.  'worst' selects the most expensive
           bucket at link rate -- the safe default when the engine gives no
           hint.

Prediction is piecewise-linear in delivered rate (GB/s) with a monotone
envelope (cummax) -- the law says cost is monotone in footprint; seed noise
must not create exploitable dips.  Inversion uses predict_upper(), which adds
a residual band max(seed IQR, margin_pp), so the admitted rate is conservative
by construction.  Pure Python: CPU-CI testable.
"""
from __future__ import annotations

import json
import bisect
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CurvePoint:
    rate_gbs: float        # delivered rate on the victim's HBM port
    victim_pct: float      # measured decode slowdown at that rate (%)
    spread_pp: float = 0.0  # across-seed spread (pp), feeds the upper band


@dataclass
class Curve:
    role: str
    route: str
    workload: str
    points: list[CurvePoint] = field(default_factory=list)
    margin_pp: float = 1.0   # floor of the conservative band

    def _sorted(self) -> list[CurvePoint]:
        return sorted(self.points, key=lambda p: p.rate_gbs)

    def _envelope(self, upper: bool) -> list[tuple[float, float]]:
        """(rate, pct) pairs, monotone non-decreasing in pct (cummax)."""
        env, hi = [], float("-inf")
        for p in self._sorted():
            y = p.victim_pct + (max(p.spread_pp, self.margin_pp) if upper else 0.0)
            hi = max(hi, y)
            env.append((p.rate_gbs, hi))
        return env

    def _interp(self, env: list[tuple[float, float]], rate: float) -> float:
        if not env:
            raise ValueError(f"empty curve {self.role}/{self.route}/{self.workload}")
        xs = [x for x, _ in env]
        if rate <= xs[0]:
            # below the lowest calibrated rate: scale down linearly through 0
            # (cost at zero footprint is zero by definition)
            return env[0][1] * rate / xs[0] if xs[0] > 0 else env[0][1]
        if rate >= xs[-1]:
            return env[-1][1]   # clamp: never extrapolate past the census
        i = bisect.bisect_right(xs, rate)
        (x0, y0), (x1, y1) = env[i - 1], env[i]
        return y0 + (rate - x0) / (x1 - x0) * (y1 - y0)

    def predict(self, rate_gbs: float) -> float:
        """Central estimate of victim slowdown (%) at delivered rate."""
        return self._interp(self._envelope(upper=False), max(0.0, rate_gbs))

    def predict_upper(self, rate_gbs: float) -> float:
        """Conservative (envelope + residual band) estimate, used to admit."""
        return self._interp(self._envelope(upper=True), max(0.0, rate_gbs))

    def invert(self, eps_pct: float) -> float:
        """Max delivered rate (GB/s) with predict_upper(rate) <= eps_pct.

        Walks the upper envelope; inside a segment solves linearly.  Returns
        0.0 if even rate->0 violates (cannot happen: cost(0)=0), and the max
        calibrated rate if eps is above the whole curve.
        """
        env = self._envelope(upper=True)
        if not env:
            raise ValueError("empty curve")
        if eps_pct >= env[-1][1]:
            return env[-1][0]
        # virtual origin (0, 0)
        prev_x, prev_y = 0.0, 0.0
        for x, y in env:
            if y >= eps_pct:
                if y == prev_y:
                    return prev_x
                return prev_x + (eps_pct - prev_y) / (y - prev_y) * (x - prev_x)
            prev_x, prev_y = x, y
        return env[-1][0]


@dataclass
class GovernorCalibration:
    device: str = ""
    hbm_peak_gbs: float = 0.0
    link_peak_gbs: dict[str, float] = field(default_factory=dict)  # route -> GB/s
    baseline_ms: dict[str, float] = field(default_factory=dict)    # workload -> ms
    curves: dict[str, Curve] = field(default_factory=dict)         # "role/route/workload"
    meta: dict = field(default_factory=dict)
    _synth: dict = field(default_factory=dict, repr=False)         # cache, not saved

    @staticmethod
    def key(role: str, route: str, workload: str) -> str:
        return f"{role}/{route}/{workload}"

    def add_curve(self, c: Curve) -> None:
        self.curves[self.key(c.role, c.route, c.workload)] = c
        self._synth.clear()      # the cached pointwise-max envelope is stale

    def workloads(self, role: str, route: str) -> list[str]:
        pre = f"{role}/{route}/"
        return [k.split("/")[2] for k in self.curves if k.startswith(pre)]

    def curve(self, role: str, route: str, workload: str = "worst") -> Curve:
        """'worst' = POINTWISE max across workload buckets.

        Buckets cross: on the A100 census, b8 costs more than b1 at mid
        rates but less at link rate, so "pick the bucket that is worst at
        link rate" admits rates that violate eps on the other bucket
        (measured: governor-ff at the b1-derived cap ran 72-86% violation
        epochs on a b8 victim).  The sound hint-less envelope is the max
        over buckets at EVERY rate; inversion against it is safe for any
        victim the census covered.
        """
        if workload != "worst":
            k = self.key(role, route, workload)
            if k in self.curves:
                return self.curves[k]
            workload = "worst"          # unknown hint: fall through, stay safe
        cands = [self.curves[self.key(role, route, w)]
                 for w in self.workloads(role, route)]
        if not cands:
            raise KeyError(f"no curves for {role}/{route}")
        if len(cands) == 1:
            return cands[0]
        ck = self.key(role, route, "__worst__")
        if ck not in self._synth:
            rates = sorted({p.rate_gbs for c in cands for p in c.points})
            pts = []
            for r in rates:
                lo = max(c.predict(r) for c in cands)
                hi = max(c.predict_upper(r) for c in cands)
                pts.append(CurvePoint(rate_gbs=r, victim_pct=lo,
                                      spread_pp=hi - lo))
            self._synth[ck] = Curve(role=role, route=route, workload="worst",
                                    points=pts, margin_pp=0.0)
        return self._synth[ck]

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        blob = {
            "device": self.device,
            "hbm_peak_gbs": self.hbm_peak_gbs,
            "link_peak_gbs": self.link_peak_gbs,
            "baseline_ms": self.baseline_ms,
            "meta": self.meta,
            "curves": {k: {**asdict(c), "points": [asdict(p) for p in c.points]}
                       for k, c in self.curves.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GovernorCalibration":
        blob = json.loads(Path(path).read_text())
        cal = cls(device=blob.get("device", ""),
                  hbm_peak_gbs=blob.get("hbm_peak_gbs", 0.0),
                  link_peak_gbs=blob.get("link_peak_gbs", {}),
                  baseline_ms=blob.get("baseline_ms", {}),
                  meta=blob.get("meta", {}))
        for k, c in blob.get("curves", {}).items():
            pts = [CurvePoint(**p) for p in c.pop("points")]
            cal.curves[k] = Curve(points=pts, **{kk: vv for kk, vv in c.items()
                                                 if kk in ("role", "route",
                                                           "workload", "margin_pp")})
        return cal

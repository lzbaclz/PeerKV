"""BudgetTrim -- the slow closed loop (IOCost-style vrate trim).

Transfers complete in single-digit milliseconds; no feedback loop can act
*within* one (critique F6).  So the calibrated feedforward curve carries the
fast path, and this loop corrects the NEXT admissions: every epoch it compares
the victim's engine-reported iteration times against an auto-tracked clean
baseline and multiplies the ledger's inverted budget by m (AIMD):

    slowdown > eps          -> m *= max(0.5, 0.9 * eps / slowdown)  (back off)
    slowdown < probe_frac*eps and work pending -> m *= up_gain      (probe up)

The clean baseline is learned online: epochs during which the ledger held no
active lease (and ambient was quiet) update an EMA of the victim's iteration
median.  This avoids any oracle baseline -- the controller never needs to be
told what "alone" looks like (closes the circularity objection).

Pure Python; the caller (Governor) drives epochs and feeds samples.
"""
from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field


@dataclass
class TrimState:
    multiplier: float = 1.0
    baseline_ms: float | None = None
    last_slowdown_pct: float | None = None
    epochs: int = 0
    backoffs: int = 0
    probes: int = 0
    history: list = field(default_factory=list)   # (epoch, m, slowdown_pct)


class BudgetTrim:
    def __init__(self, eps_pct: float,
                 m0: float = 1.0, m_min: float = 0.05, m_max: float = 16.0,
                 probe_frac: float = 0.5, up_gain: float = 1.15,
                 baseline_ema: float = 0.2, keep_history: int = 4096):
        self.eps_pct = eps_pct
        self.m_min, self.m_max = m_min, m_max
        self.probe_frac, self.up_gain = probe_frac, up_gain
        self.baseline_ema = baseline_ema
        self.keep_history = keep_history
        self._lock = threading.Lock()
        self.state = TrimState(multiplier=m0)

    @property
    def multiplier(self) -> float:
        with self._lock:
            return self.state.multiplier

    def observe_idle(self, victim_ms_samples: list[float]) -> None:
        """Epoch with no governed traffic: refresh the clean baseline."""
        if not victim_ms_samples:
            return
        med = statistics.median(victim_ms_samples)
        with self._lock:
            st = self.state
            st.baseline_ms = (med if st.baseline_ms is None else
                              (1 - self.baseline_ema) * st.baseline_ms
                              + self.baseline_ema * med)

    def observe_starved(self) -> float:
        """Loaded epoch whose previously-alive victim feed went SILENT.

        A vanished feed while our traffic is in flight is the victim-collapse
        signature (measured: a real vLLM engine's token completions stall
        wholesale past a burst threshold -- no completions, no TPOT samples,
        and a naive controller freezes at the harmful rate).  Missing data is
        danger, not absence of information: back off as if eps were doubled.
        """
        with self._lock:
            st = self.state
            st.epochs += 1
            st.backoffs += 1
            st.multiplier = max(self.m_min, st.multiplier * 0.5)
            st.last_slowdown_pct = None
            return st.multiplier

    def observe_loaded(self, victim_ms_samples: list[float],
                       work_pending: bool) -> float:
        """Epoch with governed traffic in flight: trim.  Returns multiplier."""
        with self._lock:
            st = self.state
            st.epochs += 1
            if not victim_ms_samples or st.baseline_ms is None:
                return st.multiplier        # nothing to learn from yet
            med = statistics.median(victim_ms_samples)
            slow = (med / st.baseline_ms - 1.0) * 100.0
            st.last_slowdown_pct = slow
            if slow > self.eps_pct:
                st.multiplier *= max(0.5, 0.9 * self.eps_pct / max(slow, 1e-9))
                st.backoffs += 1
            elif slow < self.probe_frac * self.eps_pct and work_pending:
                st.multiplier *= self.up_gain
                st.probes += 1
            st.multiplier = min(self.m_max, max(self.m_min, st.multiplier))
            if len(st.history) < self.keep_history:
                st.history.append((st.epochs, round(st.multiplier, 4),
                                   round(slow, 3)))
            return st.multiplier

    def snapshot(self) -> dict:
        with self._lock:
            st = self.state
            return {"multiplier": round(st.multiplier, 4),
                    "baseline_ms": st.baseline_ms,
                    "last_slowdown_pct": st.last_slowdown_pct,
                    "epochs": st.epochs, "backoffs": st.backoffs,
                    "probes": st.probes}

"""PeerKV Governor -- interference-priced KV movement (the cost model, inverted).

The ICCD measurement paper established the forward law: a concurrent copy's
victim cost is a continuous monotone of the bandwidth footprint it places on
the victim's HBM port, with write polarity (receiver) costing ~2x read polarity
(holder) at matched rate, and footprints of concurrent transfers adding (g8).
This package inverts it: given a do-no-harm budget eps on each endpoint,
solve for the maximum admissible delivered rate, then *enforce* that rate with
chunk-granular pacing -- the only actuation knob that works (g9: stream
priority is a null; direction is a null).

Components (two-timescale control, IOCost-style):
  calib      -- footprint->victim-cost curves fitted from a REAL-COPY census
                (s1), per (role in {holder,receiver}, route in {peer,host},
                workload bucket); invertible with a conservative residual band.
  budget     -- Ledger: dual-ended rate leases with additive admission
                (aggregate lease rate per endpoint+polarity goes through the
                curve; new lease gets what is left under eps).
  pacer      -- PacedCopier: one transfer as a paced chunk train on a dedicated
                stream; the per-chunk inter-launch gap realizes the lease rate.
  control    -- BudgetTrim: slow AIMD loop on engine-reported victim iteration
                times; multiplies the ledger's admissible rate (feedforward is
                the fast path -- transfers finish in ms, feedback cannot act
                within one; it corrects the *next* ones).
  estimator  -- HeadroomEstimator: NVML poller (NVLink TX/RX counters work
                without root) for ambient traffic the ledger did not admit.
  governor   -- facade: quote() / submit() / feed_victim() / report().

CPU-importable: calib/budget/control are pure Python (CI-testable); torch is
imported lazily inside pacer/governor; pynvml degrades to a zero estimator.
"""
from .calib import Curve, CurvePoint, GovernorCalibration
from .budget import Ledger, Lease
from .control import BudgetTrim
from .estimator import HeadroomEstimator
from .governor import Governor, TransferRequest, TransferHandle

__all__ = [
    "Curve", "CurvePoint", "GovernorCalibration",
    "Ledger", "Lease", "BudgetTrim", "HeadroomEstimator",
    "Governor", "TransferRequest", "TransferHandle",
]

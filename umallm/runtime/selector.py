"""Online do-no-harm corner selector (Track C P2 / Track B E-B5).

Implements the three work items the previous stub listed:
  (a) drive ``select_point`` from LIVE peer/link state (NVML utilization +
      free HBM on the peer, cached topology/NVLink health),
  (b) the single-exit discipline: every decision leaves through
      ``elastic_policy.enforce_do_no_harm`` (R1-fit / R1-busy / R1-route /
      R1-tp-oracle) -- by construction the candidate set already respects the
      invariants, so the enforce call is a tripwire, not a filter,
  (c) emit the 04_cross_cutting SS2.2 metrics (selected point, admissible set
      size, violations, peer-busy, link health).

Modes (PEERKV_STRICT env or ``strict=`` kwarg):
  strict  : a violation raises DoNoHarmViolation (CI / tests).
  repair  : production default -- the decision is coerced to the safest legal
            corner, the violation counter increments, a WARN is logged.

CPU-safe: without pynvml/torch the live probe degrades to the constructor
defaults so unit tests can inject synthetic PeerState/LinkState.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace

from ..elastic_policy import (
    Deployment, Decision, DecodeStepModel, DoNoHarmViolation, Geometry,
    LinkState, OperatingPoint, PeerState, _single_capacity_tokens,
    admissible_points, enforce_do_no_harm, select_point,
)
from ..observability import metrics as _m

logger = logging.getLogger("peerkv.selector")

STATUS = "IMPLEMENTED: online selector with do-no-harm single-exit (Track C P2)"


def _strict_default() -> bool:
    return os.environ.get("PEERKV_STRICT", "0") == "1"


@dataclass
class LiveBoxProbe:
    """Live peer/link state source. NVML for utilization + free HBM (cheap,
    per-call); topology (NVLink health) probed once and cached with a TTL --
    `nvidia-smi topo` is a subprocess, too slow for the hot path."""
    peer_index: int = 1
    busy_util_threshold: float = 10.0      # % SM util above which peer "computes"
    topology_ttl_s: float = 30.0
    nvlink_eff_gbps: float = LinkState.nvlink_eff_gbps
    pcie_eff_gbps: float = LinkState.pcie_eff_gbps

    _topo_checked_at: float = 0.0
    _nvlink_ok: bool = True

    def _refresh_topology(self) -> None:
        now = time.monotonic()
        if now - self._topo_checked_at < self.topology_ttl_s:
            return
        self._topo_checked_at = now
        try:
            from ..observability.topology_probe import probe_topology
            topo = probe_topology()
            # probe_topology's verdict key is peer_mode_ok (MIG off + NVLink
            # trained + P2P) -- a missing key means the probe could not decide,
            # which must NOT silently read as healthy.
            self._nvlink_ok = bool(topo.get("peer_mode_ok", False)) \
                if topo else True
        except Exception:                      # noqa: BLE001 - degrade, don't die
            self._nvlink_ok = True

    def peer_state(self) -> PeerState:
        """Live peer state. Failure semantics: pynvml ABSENT (CPU sandbox /
        unit tests) degrades to optimistic defaults; pynvml PRESENT but
        erroring (driver mismatch, container without /dev/nvidiactl) degrades
        PESSIMISTIC -- a broken probe on a real GPU box must not admit
        CFK/COPYBACK against an unknown peer (do-no-harm)."""
        try:
            import pynvml
        except Exception:                      # noqa: BLE001 - CPU sandbox
            idle, free = True, 70 * 1024 ** 3
        else:
            try:
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(self.peer_index)
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                idle = util < self.busy_util_threshold
                free = int(mem.free)
            except Exception:                  # noqa: BLE001 - NVML errored
                logger.warning("NVML probe failed on a GPU box: assuming "
                               "peer BUSY with 0 free HBM (pessimistic)")
                idle, free = False, 0
        self._refresh_topology()
        bw = self.nvlink_eff_gbps if self._nvlink_ok else 0.0
        # publish the link state for the peerkv_attn hot-path tripwire
        os.environ["PEERKV_NVLINK_GBPS"] = str(bw)
        _m.PEER_BUSY.labels(peer=str(self.peer_index)).set(0 if idle else 1)
        _m.LINK_EFF_GBPS.labels(fabric="nvlink").set(bw)
        return PeerState(hbm_free_bytes=free, compute_idle=idle, nvlink_bw_gbps=bw)

    def link_state(self, peer: PeerState) -> LinkState:
        return LinkState(nvlink_eff_gbps=peer.nvlink_bw_gbps,
                         pcie_eff_gbps=self.pcie_eff_gbps)


def _r1_candidates(ctx_tokens: int, geom: Geometry, peer: PeerState,
                   link: LinkState, deploy: Deployment) -> set:
    """Admissible set restricted so the invariants hold BY CONSTRUCTION.
    enforce_do_no_harm afterwards is then a pure tripwire."""
    adm = admissible_points(ctx_tokens, geom, peer, deploy)
    cap = _single_capacity_tokens(geom, deploy)
    if ctx_tokens <= cap:
        return {OperatingPoint.SINGLE} if OperatingPoint.SINGLE in adm else adm
    if link.nvlink_eff_gbps < link.pcie_eff_gbps:
        adm -= {OperatingPoint.CFK, OperatingPoint.COPYBACK}
    if not peer.compute_idle:
        adm -= {OperatingPoint.CFK}
    # TP stays in the set for the admissibility record; without a measured
    # TPOT its prediction is inf so it can never win the argmin (R3).
    return adm


def online_select(ctx_tokens: int, geom: Geometry, deploy: Deployment,
                  model: DecodeStepModel,
                  deadline_ms: "float | None" = None,
                  peer: "PeerState | None" = None,
                  link: "LinkState | None" = None,
                  probe: "LiveBoxProbe | None" = None,
                  strict: "bool | None" = None) -> Decision:
    """The single exit of the selector: live state -> R1-restricted candidate
    set -> deadline-gated argmin -> enforce_do_no_harm tripwire -> metrics."""
    strict = _strict_default() if strict is None else strict
    if peer is None:
        probe = probe or LiveBoxProbe()
        peer = probe.peer_state()
    if link is None:
        link = (probe.link_state(peer) if probe is not None
                else LinkState.from_peer(peer))

    candidates = _r1_candidates(ctx_tokens, geom, peer, link, deploy)
    decision = select_point(ctx_tokens, geom, peer, deploy, model, deadline_ms)
    if decision.point not in candidates and candidates:
        # select_point chose outside the R1 set (e.g. measured-TP faster than
        # SINGLE on a fitting request): re-pick inside the legal set.
        preds = {p: model.predict_ms(p, ctx_tokens, peer, deploy)
                 for p in candidates}
        best = min(preds, key=preds.get)
        meets = deadline_ms is None or preds[best] <= deadline_ms
        decision = replace(decision, point=best, predicted_ms=preds[best],
                           admissible=candidates, meets_deadline=meets,
                           reason=decision.reason + "; R1-restricted")

    try:
        enforce_do_no_harm(decision.point, ctx_tokens, geom, peer, link,
                           deploy=deploy)
    except DoNoHarmViolation as e:
        rule = str(e).split(":", 1)[0]
        _m.DO_NO_HARM_VIOL.labels(rule=rule).inc()
        logger.warning("do-no-harm violation: %s", e)
        if strict:
            raise
        decision = _repair(decision, ctx_tokens, geom, peer, link, deploy,
                           model, rule, deadline_ms)

    _m.SELECTED_POINT.labels(point=decision.point.value).inc()
    _m.ADMISSIBLE_POINTS.set(len(decision.admissible))
    return decision


def _repair(decision: Decision, ctx_tokens: int, geom: Geometry,
            peer: PeerState, link: LinkState, deploy: Deployment,
            model: DecodeStepModel, rule: str,
            deadline_ms: "float | None" = None) -> Decision:
    """Production fallback: coerce to the safest legal corner. The repaired
    point is re-asserted through enforce_do_no_harm (defense-in-depth: a
    future edit to _r1_candidates must not be able to break this silently)."""
    cap = _single_capacity_tokens(geom, deploy)
    if ctx_tokens <= cap:
        point = OperatingPoint.SINGLE
    else:
        legal = _r1_candidates(ctx_tokens, geom, peer, link, deploy)
        legal.discard(decision.point)
        point = (min(legal, key=lambda p: model.predict_ms(p, ctx_tokens, peer, deploy))
                 if legal else OperatingPoint.HOST)
    enforce_do_no_harm(point, ctx_tokens, geom, peer, link, deploy=deploy)
    pred = model.predict_ms(point, ctx_tokens, peer, deploy)
    meets = deadline_ms is None or pred <= deadline_ms
    return replace(decision, point=point, predicted_ms=pred,
                   meets_deadline=meets,
                   reason=f"{decision.reason}; repaired({rule})->{point.value}")

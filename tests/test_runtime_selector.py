"""Unit tests for the online selector (umallm/runtime/selector.py)."""
from __future__ import annotations

import pytest

from umallm.elastic_policy import (
    Deployment, DecodeStepModel, DoNoHarmViolation, Geometry, LinkState,
    OperatingPoint, PeerState, _single_capacity_tokens)
from umallm.runtime.selector import LiveBoxProbe, online_select

GEOM = Geometry.llama2_7b_mha()
MODEL = DecodeStepModel.calibrate(
    GEOM, single_pts={16384: 17.3, 32768: 22.2},
    cfk_pt=(32768, 25.1), copyback_pt=(32768, 65.5))
DEPLOY = Deployment()
CAP = _single_capacity_tokens(GEOM, DEPLOY)
OOM_CTX = CAP + 50_000


def test_overflow_idle_peer_prefers_cfk():
    d = online_select(OOM_CTX, GEOM, DEPLOY, MODEL, peer=PeerState(),
                      link=LinkState(), strict=True)
    assert d.point is OperatingPoint.CFK
    assert d.predicted_ms < float("inf")


def test_overflow_busy_peer_falls_to_copyback():
    d = online_select(OOM_CTX, GEOM, DEPLOY, MODEL,
                      peer=PeerState(compute_idle=False),
                      link=LinkState(), strict=True)
    assert d.point is OperatingPoint.COPYBACK
    assert "R1" not in d.reason or "restricted" in d.reason


def test_measured_tp_never_beats_single_on_fitting_request():
    """R1-fit dominates even when a measured TP TPOT is numerically faster."""
    deploy = Deployment(tp_enabled=True, tp_tpot_ms_per_token=1.0)  # absurdly fast
    ctx = 32_768                                  # fits single
    d = online_select(ctx, GEOM, deploy, MODEL, peer=PeerState(),
                      link=LinkState(), strict=True)
    assert d.point is OperatingPoint.SINGLE


def test_strict_mode_raises_when_no_legal_repair_needed():
    # sanity: strict never raises on a well-formed path
    online_select(1024, GEOM, DEPLOY, MODEL, peer=PeerState(),
                  link=LinkState(), strict=True)


def test_repair_mode_counts_and_coerces(monkeypatch):
    """Force a violation through a poisoned candidate set and confirm repair."""
    from umallm.runtime import selector as S
    monkeypatch.setattr(S, "_r1_candidates",
                        lambda *a, **k: {OperatingPoint.CFK})
    d = online_select(1024, GEOM, DEPLOY, MODEL,
                      peer=PeerState(compute_idle=False),
                      link=LinkState(), strict=False)
    assert d.point is OperatingPoint.SINGLE          # repaired to safest
    assert "repaired" in d.reason
    with pytest.raises(DoNoHarmViolation):
        online_select(1024, GEOM, DEPLOY, MODEL,
                      peer=PeerState(compute_idle=False),
                      link=LinkState(), strict=True)


def test_repair_overflow_branch(monkeypatch):
    """Poisoned candidates on an OVERFLOW request: repair must land on a
    corner that is legal for the live peer/link (busy => not CFK), and the
    repaired decision must itself pass the enforce gate (defense-in-depth
    re-assert inside _repair)."""
    from umallm.runtime import selector as S
    real = S._r1_candidates
    calls = {"n": 0}

    def poisoned(*a, **k):
        calls["n"] += 1
        # first call (selection) returns the illegal CFK-only set; the
        # repair path's second call sees the real legal set
        return {OperatingPoint.CFK} if calls["n"] == 1 else real(*a, **k)

    monkeypatch.setattr(S, "_r1_candidates", poisoned)
    d = online_select(OOM_CTX, GEOM, DEPLOY, MODEL,
                      peer=PeerState(compute_idle=False),
                      link=LinkState(), strict=False)
    assert d.point in {OperatingPoint.COPYBACK, OperatingPoint.HOST}
    assert "repaired(R1-busy)" in d.reason


def test_deadline_recomputed_after_repick():
    """meets_deadline must reflect the FINAL point, not the pre-coercion
    argmin (audit: stale meets_deadline bug)."""
    deploy = Deployment(tp_enabled=True, tp_tpot_ms_per_token=1.0)
    ctx = 32_768                                   # fits single
    # deadline between TP's 1.0ms and single's ~22ms: the re-pick to SINGLE
    # must flip meets_deadline to False.
    d = online_select(ctx, GEOM, deploy, MODEL, peer=PeerState(),
                      link=LinkState(), deadline_ms=5.0, strict=True)
    assert d.point is OperatingPoint.SINGLE
    assert d.meets_deadline is False


def test_probe_degrades_without_gpu():
    p = LiveBoxProbe()
    st = p.peer_state()       # must not raise in a CPU sandbox
    assert st.hbm_free_bytes > 0
    link = p.link_state(st)
    assert link.pcie_eff_gbps > 0

"""The four do-no-harm CI tests (spec: collaboration_plan/04_cross_cutting.md SS1.3).

  (a) test_no_harm_single_fits     fits-single => SINGLE, always; gate really trips
  (b) test_no_harm_peer_busy       peer computing => CFK never admissible/chosen
  (c) test_no_harm_nvlink_degraded NVLink < PCIe => peer paths closed, host routes
  (d) test_no_harm_ab_vs_vllm      GPU A/B vs vanilla vLLM (@pytest.mark.gpu)

(a)-(c) are pure CPU and run in every CI push; (d) needs the dual-A100 box.
These tests are the paper's R1 red line turned into executable gates (D3).
"""
from __future__ import annotations

import json
import os
import random

import pytest

from umallm.elastic_policy import (
    BETA_NVLINK_GBPS, BETA_PCIE_GBPS, LENDER_FLOPS_RETAINED,
    Deployment, DecodeStepModel, DoNoHarmViolation, Geometry, LinkState,
    OperatingPoint, PeerState, _single_capacity_tokens, admissible_points,
    enforce_do_no_harm, select_point,
)
from umallm.runtime.selector import online_select

# --- a realistic calibrated model (e27 real-weights anchors, MHA) ---------- #
GEOM = Geometry.llama2_7b_mha()
MODEL = DecodeStepModel.calibrate(
    GEOM, single_pts={16384: 17.3, 32768: 22.2},
    cfk_pt=(32768, 25.1), copyback_pt=(32768, 65.5))
DEPLOY = Deployment()                 # measured caps: single 116016, TP 253536
CAP = _single_capacity_tokens(GEOM, DEPLOY)
HEALTHY = LinkState()                 # 273 vs 24 GB/s
IDLE = PeerState()
BUSY = PeerState(compute_idle=False)


def _random_peers(n: int, seed: int = 7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(PeerState(
            hbm_free_bytes=rng.choice([0, 8, 35, 70]) * 1024 ** 3,
            compute_idle=rng.random() < 0.5,
            nvlink_bw_gbps=rng.choice([BETA_NVLINK_GBPS, BETA_NVLINK_GBPS, 100.0])))
    return out


# --------------------------------------------------------------------------- #
# (a) R1-fit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ctx", [1, 1024, CAP - 1, CAP])
def test_no_harm_single_fits(ctx):
    for peer in _random_peers(5):
        d = online_select(ctx, GEOM, DEPLOY, MODEL, peer=peer,
                          link=LinkState.from_peer(peer), strict=True)
        assert d.point is OperatingPoint.SINGLE, (
            f"ctx={ctx} fits single but selector chose {d.point}")
        # and the gate itself blesses it
        assert enforce_do_no_harm(d.point, ctx, GEOM, peer,
                                  LinkState.from_peer(peer), deploy=DEPLOY) \
            is OperatingPoint.SINGLE


def test_no_harm_single_fits_gate_trips():
    """Adversarial fuzz: a selector that returns CFK for a fitting request
    must be caught -- proves the gate is not a no-op."""
    ctx = 1024
    with pytest.raises(DoNoHarmViolation, match="R1-fit"):
        enforce_do_no_harm(OperatingPoint.CFK, ctx, GEOM, IDLE, HEALTHY,
                           deploy=DEPLOY)
    # adversarial model: online_select neutralizes it via the R1 candidate
    # re-pick (first line of defense; enforce_do_no_harm is the tripwire
    # behind it -- exercised by tests/test_runtime_selector.py's poisoned-
    # candidate tests).
    class _EvilModel(DecodeStepModel):
        def predict_ms(self, point, ctx_tokens, peer, deploy):
            # pretend CFK is always fastest
            return 0.0 if point is OperatingPoint.CFK else 100.0
    evil = _EvilModel(GEOM, MODEL.A_ms, MODEL.beta_eff_gbps,
                      MODEL.roundtrip_ms_per_layer)
    d = online_select(ctx, GEOM, DEPLOY, evil, peer=IDLE, link=HEALTHY,
                      strict=False)
    assert d.point is OperatingPoint.SINGLE


# --------------------------------------------------------------------------- #
# (b) R1-busy
# --------------------------------------------------------------------------- #
def test_no_harm_peer_busy():
    ctx = CAP + 50_000          # force off-card
    # (i) CFK not admissible
    adm = admissible_points(ctx, GEOM, BUSY, DEPLOY)
    assert OperatingPoint.CFK not in adm
    # (ii) selector lands in the legal complement
    d = select_point(ctx, GEOM, BUSY, DEPLOY, MODEL)
    assert d.point in {OperatingPoint.COPYBACK, OperatingPoint.HOST,
                       OperatingPoint.TP, OperatingPoint.INFEASIBLE}
    # (iii) the gate trips on a forced CFK
    with pytest.raises(DoNoHarmViolation, match="R1-busy"):
        enforce_do_no_harm(OperatingPoint.CFK, ctx, GEOM, BUSY, HEALTHY,
                           deploy=DEPLOY)
    # numeric accounting: the rule encodes the measured e23 physics, not a
    # magic constant -- the lender keeps only ~2/3 of its FLOPs.
    assert LENDER_FLOPS_RETAINED == pytest.approx(0.6676, abs=1e-4)
    assert 1.0 - LENDER_FLOPS_RETAINED > 0.30


# --------------------------------------------------------------------------- #
# (c) R1-route
# --------------------------------------------------------------------------- #
def test_no_harm_nvlink_degraded():
    ctx = CAP + 50_000
    degraded_peer = PeerState(nvlink_bw_gbps=20.0)        # < PCIe 24
    degraded_link = LinkState(nvlink_eff_gbps=20.0, pcie_eff_gbps=24.0)
    # (i)+(ii) selector avoids peer-NVLink paths and routes host
    d = online_select(ctx, GEOM, DEPLOY, MODEL, peer=degraded_peer,
                      link=degraded_link, strict=True)
    assert d.point not in {OperatingPoint.CFK, OperatingPoint.COPYBACK}
    assert d.point is OperatingPoint.HOST
    # gate trips on a forced peer path
    for bad in (OperatingPoint.CFK, OperatingPoint.COPYBACK):
        with pytest.raises(DoNoHarmViolation, match="R1-route"):
            enforce_do_no_harm(bad, ctx, GEOM, degraded_peer, degraded_link,
                               deploy=DEPLOY)
    # (iii) healthy link admits the peer paths again
    healthy_peer = PeerState()
    assert enforce_do_no_harm(OperatingPoint.COPYBACK, ctx, GEOM,
                              healthy_peer, HEALTHY, deploy=DEPLOY) \
        is OperatingPoint.COPYBACK
    # direction is recorded, NOT constrained (Track A: push/pull equivalent)
    assert enforce_do_no_harm(
        OperatingPoint.COPYBACK, ctx, GEOM, healthy_peer,
        LinkState(transfer_dir="pull"), deploy=DEPLOY) is OperatingPoint.COPYBACK


# --------------------------------------------------------------------------- #
# R3: TP is admissibility-only without a measured TPOT
# --------------------------------------------------------------------------- #
def test_no_harm_tp_admissibility_only():
    ctx = CAP + 50_000
    deploy_tp = Deployment(tp_enabled=True)               # no measured TPOT
    assert MODEL.predict_ms(OperatingPoint.TP, ctx, IDLE, deploy_tp) == float("inf")
    d = select_point(ctx, GEOM, IDLE, deploy_tp, MODEL)
    assert d.point is not OperatingPoint.TP
    with pytest.raises(DoNoHarmViolation, match="R1-tp-oracle"):
        enforce_do_no_harm(OperatingPoint.TP, ctx, GEOM, IDLE, HEALTHY,
                           deploy=deploy_tp)
    # the analytic weight-read bound EXISTS (executed, not a comment) but is
    # a diagnostic: finite, and never consulted by predict_ms.
    bound = MODEL.predict_tp_bound_ms(ctx)
    assert 0.0 < bound < float("inf")
    # with a measured TPOT, TP becomes a legal latency point
    deploy_meas = Deployment(tp_enabled=True, tp_tpot_ms_per_token=20.6)
    assert MODEL.predict_ms(OperatingPoint.TP, ctx, IDLE, deploy_meas) == 20.6
    assert enforce_do_no_harm(OperatingPoint.TP, ctx, GEOM, IDLE, HEALTHY,
                              deploy=deploy_meas) is OperatingPoint.TP


# --------------------------------------------------------------------------- #
# (d) GPU A/B vs vanilla vLLM -- the measured R1 evidence
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_no_harm_ab_vs_vllm():
    """Fitting-context workload: PeerKV-enabled serving must not be slower
    than vanilla vLLM (TPOT/TTFT medians within EPS). Skips unless the dual-
    A100 box, vLLM, and an idle peer are all available."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    pytest.importorskip("vllm")
    from umallm.observability.box_probe import box_idle
    idle, detail = box_idle()
    if not idle:
        pytest.skip(f"box busy (co-tenant): {detail}")

    art = os.environ.get("PEERKV_AB_ARTIFACT",
                         "tests/_artifacts/ab_vs_vllm.json")
    if not os.path.exists(art):
        pytest.skip("A/B artifact not produced yet -- run "
                    "experiments/eB1_ab_vs_vllm.py first (M2 gate)")
    data = json.load(open(art))
    eps = data.get("eps", 0.03)
    assert data["tpot_p50_peerkv_ms"] <= data["tpot_p50_vllm_ms"] * (1 + eps)
    assert data["ttft_p50_peerkv_ms"] <= data["ttft_p50_vllm_ms"] * (1 + eps)
    assert data.get("do_no_harm_violations", 0) == 0

"""Tests for the cost-model-gated elastic-parallelism policy (CPU, no GPU)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umallm.elastic_policy import (  # noqa: E402
    BETA_PCIE_GBPS, Deployment, DecodeStepModel, Geometry, OperatingPoint,
    PeerState, admissible_points, select_point,
)


def _mha_model():
    geom = Geometry.llama2_7b_mha()
    # calibrate from the committed e27/e31 MHA numbers
    return geom, DecodeStepModel.calibrate(
        geom, single_pts={16384: 17.257, 32768: 22.151},
        cfk_pt=(16384, 20.336), copyback_pt=(16384, 38.402))


def test_admissibility_fit_vs_oom():
    geom = Geometry.llama2_7b_mha()
    dep = Deployment()
    fit = admissible_points(16384, geom, PeerState(), dep)
    assert OperatingPoint.SINGLE in fit
    oom = admissible_points(143360, geom, PeerState(), dep)   # > 116K ceiling
    assert OperatingPoint.SINGLE not in oom


def test_peer_busy_disables_cfk_keeps_copyback():
    geom = Geometry.llama2_7b_mha()
    dep = Deployment()
    busy = admissible_points(143360, geom, PeerState(compute_idle=False), dep)
    assert OperatingPoint.CFK not in busy
    assert OperatingPoint.COPYBACK in busy          # copy-back needs only HBM+link
    idle = admissible_points(143360, geom, PeerState(compute_idle=True), dep)
    assert OperatingPoint.CFK in idle


def test_link_degraded_routes_to_host():
    geom = Geometry.llama2_7b_mha()
    dep = Deployment()
    adm = admissible_points(143360, geom,
                            PeerState(nvlink_bw_gbps=BETA_PCIE_GBPS / 2), dep)
    assert OperatingPoint.COPYBACK not in adm and OperatingPoint.CFK not in adm
    assert OperatingPoint.HOST in adm


def test_selector_picks_single_when_fits():
    geom, model = _mha_model()
    dec = select_point(16384, geom, PeerState(), Deployment(), model)
    assert dec.point is OperatingPoint.SINGLE


def test_selector_picks_cfk_on_overflow_idle_peer():
    geom, model = _mha_model()
    dec = select_point(143360, geom, PeerState(compute_idle=True), Deployment(), model)
    assert dec.point is OperatingPoint.CFK


def test_selector_picks_copyback_on_overflow_busy_peer():
    geom, model = _mha_model()
    dec = select_point(143360, geom, PeerState(compute_idle=False), Deployment(), model)
    assert dec.point is OperatingPoint.COPYBACK


def test_cost_model_ranks_single_below_cfk_on_fitting():
    """The round-trip term must make single < CFK for a fitting context (the gate's
    whole reason to exist)."""
    geom, model = _mha_model()
    s = model.predict_ms(OperatingPoint.SINGLE, 16384, PeerState(), Deployment())
    c = model.predict_ms(OperatingPoint.CFK, 16384, PeerState(), Deployment())
    assert s < c


def test_gqa_single_capacity_larger_than_mha():
    dep = Deployment()
    from umallm.elastic_policy import _single_capacity_tokens
    assert (_single_capacity_tokens(Geometry.gqa_8kv(), dep)
            > _single_capacity_tokens(Geometry.llama2_7b_mha(), dep))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all elastic_policy tests passed")

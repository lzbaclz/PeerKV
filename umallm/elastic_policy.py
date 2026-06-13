"""Cost-model-gated elastic-parallelism policy for single-request KV overflow.

This is the *policy* layer of the repositioned PeerKV contribution (see
``docs/ELASTIC_REPOSITION.md``). A single overflowing decode request must answer
three orthogonal questions each layer:

    phi   (placement) : what fraction of KV lives off the compute GPU
    kappa (compute)   : for off-GPU KV, copy it back, or compute the partial
                        where it lives and exchange the KB-scale (O, lse)?
    W     (weights)   : weights whole on the compute GPU, or sharded (TP)?

The corner points of that (phi, kappa, W) continuum are exactly the existing
systems -- each pins itself to one corner:

    SINGLE   (phi=0,  -,         W=whole)  single-GPU, when the context fits
    COPYBACK (phi>0,  copy-back, W=whole)  Harvest (per-block) / AQUA (coalesced)
    CFK      (phi>0,  remote,    W=whole)  DistAttention / Tree (always-distribute)
    TP       (phi=1/2 by-head,   W=1/2  )  tensor parallelism (deploy-time)
    HOST     (phi>0,  copy-back, W=whole)  vertical host/PCIe offload (FlexGen)

The *new* contribution is not a new corner: it is (a) a calibrated decode-step
model that includes a **full-weight-read asymmetry term** (absent from the
single-tier SEER inversion this repo's ``min_fast_resident_for_slo`` instantiates)
which is what structurally hands TP the fitting corner, and (b) a deadline-gated
``select_point`` that picks the optimal corner per request from live peer state --
so the policy can *decline* the cross-GPU round-trip that DistAttention/Tree
always pay and *decline* the copy-back that Harvest/AQUA always pay.

The model RANKS points; it does not need to predict absolute latency precisely
(the points are well separated, see ``experiments/e33_policy_regret.py``). Pure
Python/NumPy, CPU-testable.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

# ---- measured hardware constants (dual A100-SXM4-80GB, NV12; e15/e20/e23) ---- #
BETA_HBM_GBPS = 773.0       # effective KV-block HBM bandwidth (e15)
BETA_NVLINK_GBPS = 273.0    # one-way peer NVLink (e15); NOT vendor 600 bidir
BETA_PCIE_GBPS = 24.0       # host PCIe (e15)
C_NVLINK_US = 23.6          # per-transfer setup, peer DMA (e20)  -- note > PCIe
C_PCIE_US = 12.6            # per-transfer setup, pinned H2D (e20)
C_HBM_US = 12.3
PARTIAL_BYTES_DEFAULT = 2080  # one (O fp16, lse fp32) partial, H=8 D=128 (e20)

# peer-compute contention (e23): a borrower keeps ~all its NVLink BW, but the
# lender loses ~1/3 of its matmul throughput -- so CFK (which needs the peer to
# *compute*) is only admissible when the peer is compute-idle; copy-back (which
# needs only the peer's HBM + link) stays admissible when the peer is busy.
BORROWER_BW_RETAINED = 0.9974   # e23
LENDER_FLOPS_RETAINED = 0.6676  # e23  (== loses 33%)


class OperatingPoint(enum.Enum):
    SINGLE = "single"        # phi=0, W=whole
    COPYBACK = "copyback"    # phi>0, copy-back, W=whole   (Harvest/AQUA corner)
    CFK = "compute_follows_kv"  # phi>0, remote partial, W=whole (DistAttn/Tree corner)
    TP = "tp"                # W=1/2, deploy-time         (tensor parallelism)
    HOST = "host"            # vertical PCIe offload
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class Geometry:
    """Per-token KV size and model weight bytes for one geometry."""
    name: str
    layers: int
    kv_bytes_per_token: int      # across all layers (K+V)
    weight_bytes: int            # total model weights (fp16)

    @staticmethod
    def llama2_7b_mha() -> "Geometry":
        # 32 layers x 2(KV) x 32 heads x 128 x 2B = 524288 B/token
        return Geometry("MHA", 32, 524288, int(12.0625 * 1024**3))

    @staticmethod
    def gqa_8kv() -> "Geometry":
        # 32 x 2 x 8 x 128 x 2B = 131072 B/token
        return Geometry("GQA", 32, 131072, int(13.0 * 1024**3))


@dataclass
class PeerState:
    """Live state of the peer GPU at admission time."""
    hbm_free_bytes: int = 70 * 1024**3   # KV-available HBM on the peer
    compute_idle: bool = True            # can the peer lend attention compute?
    nvlink_bw_gbps: float = BETA_NVLINK_GBPS  # 0 / <PCIe => link degraded


@dataclass
class Deployment:
    """Whether TP is even an option for this request, and the capacities."""
    tp_enabled: bool = False             # both GPUs bound to THIS model, reshardable
    single_capacity_tokens: int = 116016 # measured MHA ceiling at util0.9 (serve_m1_tp2)
    tp_capacity_tokens: int = 253536     # measured TP-2 (serve_m1_tp2)
    # optional measured single-stream TP TPOT model (ms/tok); None => admissibility-only
    tp_tpot_ms_per_token: "float | None" = None


@dataclass
class DecodeStepModel:
    """Calibrated full-model decode-step latency model (ms/token).

    step(point) = A + kv_read(local_tokens) + extra(point)
      A             : weights+compute intercept (whole-weight points). For TP the
                      weight read halves -> the **full-weight-read asymmetry term**.
      kv_read(n)    : n * kv_bytes_per_token / beta_eff  (calibrated effective HBM BW)
      extra(SINGLE) : 0                       (local = ctx)
      extra(CFK)    : layers * roundtrip      (local = ctx/2; the round-trip the
                                               always-distribute baselines can't decline)
      extra(COPYBACK): peer KV streamed back over NVLink, coalesced (local = ctx/2)
      extra(HOST)   : peer KV streamed back over PCIe

    Calibrated per geometry via :meth:`calibrate`. Constants are fit on the
    fitting points and the OOM points are held out (reported in e33/RQ1).
    """
    geom: Geometry
    A_ms: float                  # weights+compute intercept
    beta_eff_gbps: float         # effective HBM read BW for KV (calibrated)
    roundtrip_ms_per_layer: float
    copyback_eff: float = 1.0    # measured/ideal copy-back transfer ratio
    n_devices: int = 2

    # ---- physical sub-terms (GB/s, bytes, ms) ---- #
    def _kv_read_ms(self, n_tokens: float) -> float:
        return (n_tokens * self.geom.kv_bytes_per_token) / (self.beta_eff_gbps * 1e9) * 1e3

    def _copyback_ms(self, peer_tokens: float, nvlink_bw_gbps: float) -> float:
        if nvlink_bw_gbps <= 0:
            return math.inf          # degraded link: honest semantics, not a crash
        b = peer_tokens * self.geom.kv_bytes_per_token
        return self.copyback_eff * (b / (nvlink_bw_gbps * 1e9) * 1e3
                                    + self.geom.layers * C_NVLINK_US / 1e3)

    def predict_ms(self, point: OperatingPoint, ctx_tokens: int,
                   peer: PeerState, deploy: Deployment) -> float:
        local = ctx_tokens / self.n_devices
        if point is OperatingPoint.SINGLE:
            return self.A_ms + self._kv_read_ms(ctx_tokens)
        if point is OperatingPoint.CFK:
            return (self.A_ms + self._kv_read_ms(local)
                    + self.geom.layers * self.roundtrip_ms_per_layer)
        if point is OperatingPoint.COPYBACK:
            return (self.A_ms + self._kv_read_ms(local)
                    + self._copyback_ms(ctx_tokens - local, peer.nvlink_bw_gbps))
        if point is OperatingPoint.HOST:
            b = (ctx_tokens - local) * self.geom.kv_bytes_per_token
            return (self.A_ms + self._kv_read_ms(local)
                    + b / (BETA_PCIE_GBPS * 1e9) * 1e3
                    + self.geom.layers * C_PCIE_US / 1e3)
        if point is OperatingPoint.TP:
            # full-weight-read asymmetry: TP reads HALF the weights + HALF the KV.
            # A_ms folds weights+compute; we split it into a weight part (halved by
            # TP) and a residual compute part. Conservatively halve the whole
            # intercept's weight-dominated bulk only when a measured TP TPOT is
            # absent we return inf so TP is admissibility-only (honest default).
            if deploy.tp_tpot_ms_per_token is not None:
                return deploy.tp_tpot_ms_per_token
            return math.inf
        return math.inf

    def predict_tp_bound_ms(self, ctx_tokens: int,
                            allreduce_us_per_layer: float = C_NVLINK_US,
                            weight_stream_gbps: "float | None" = None) -> float:
        """The full-weight-read asymmetry term, EXECUTED (was previously only a
        comment in the TP branch of :meth:`predict_ms`).

        Analytic optimistic bound for TP-2: the weight-read share of the
        intercept halves (each GPU reads half the weights), the KV read halves
        (KV sharded by head), plus one KB-scale partial exchange per layer.

        R3 DISCIPLINE: this is a *diagnostic / map-annotation* quantity. It is
        deliberately NOT used by :meth:`predict_ms` or ``select_point`` -- TP
        remains admissibility-only (inf) until a *measured* TPOT exists, so the
        latency oracle can never be won on an analytic number. Validation
        against the measured (conservative, emulated-allreduce) e37 TPOT is
        reported by experiments/eB5; the bound is optimistic by construction
        (fused production TP sits between the two)."""
        gbps = weight_stream_gbps or self.beta_eff_gbps
        weight_ms = min(self.geom.weight_bytes / (gbps * 1e9) * 1e3, self.A_ms)
        residual_ms = self.A_ms - weight_ms
        return (residual_ms + weight_ms / 2.0
                + self._kv_read_ms(ctx_tokens / 2.0)
                + self.geom.layers * allreduce_us_per_layer / 1e3)

    @classmethod
    def calibrate(cls, geom: Geometry, single_pts: dict, cfk_pt, copyback_pt,
                  n_devices: int = 2) -> "DecodeStepModel":
        """Fit A, beta_eff from two SINGLE points; roundtrip from one CFK point;
        copyback_eff from one COPYBACK point. ``single_pts`` = {ctx: ms}, must
        have >=2 entries. ``cfk_pt`` / ``copyback_pt`` = (ctx, ms)."""
        if len(single_pts) < 2:
            raise ValueError("calibrate needs >=2 SINGLE anchors")
        (c1, m1), (c2, m2) = sorted(single_pts.items())[:2]
        if c1 == c2:
            raise ValueError(f"degenerate SINGLE anchors: equal ctx {c1}")
        # m = A + c*kvB/beta_eff  -> slope s = (m2-m1)/(c2-c1) ms per token
        s = (m2 - m1) / (c2 - c1)
        if s <= 0:
            raise ValueError(
                f"non-physical anchors: latency not increasing with context "
                f"(slope {s:.3e} ms/token from {single_pts}); refusing to fit "
                f"a negative beta_eff -- remeasure (noisy/co-tenant run?)")
        beta_eff = geom.kv_bytes_per_token / (s * 1e-3) / 1e9  # GB/s
        A = m1 - s * c1
        self = cls(geom, A, beta_eff, roundtrip_ms_per_layer=0.0, n_devices=n_devices)
        cctx, cms = cfk_pt
        local = cctx / n_devices
        self.roundtrip_ms_per_layer = max(
            0.0, (cms - A - self._kv_read_ms(local)) / geom.layers)
        pctx, pms = copyback_pt
        plocal = pctx / n_devices
        ideal = (self._kv_read_ms(plocal)
                 + (pctx - plocal) * geom.kv_bytes_per_token / (BETA_NVLINK_GBPS * 1e9) * 1e3
                 + geom.layers * C_NVLINK_US / 1e3)
        self.copyback_eff = max(1.0, (pms - A) / ideal) if ideal > 0 else 1.0
        return self


def admissible_points(ctx_tokens: int, geom: Geometry, peer: PeerState,
                      deploy: Deployment) -> set:
    """Which corners are *available* for this request right now."""
    pts = set()
    # weight bytes + KV bytes must fit one GPU for SINGLE
    single_cap = _single_capacity_tokens(geom, deploy)
    if ctx_tokens <= single_cap:
        pts.add(OperatingPoint.SINGLE)
    peer_kv_fits = (ctx_tokens - ctx_tokens // 2) * geom.kv_bytes_per_token <= peer.hbm_free_bytes
    # link must be at least as fast as host to be worth using as the near tier;
    # if NVLink degrades below PCIe (MIG/untrained), spill routes to host instead.
    link_ok = peer.nvlink_bw_gbps >= BETA_PCIE_GBPS
    if peer_kv_fits and link_ok:
        pts.add(OperatingPoint.COPYBACK)        # needs only peer HBM + link
        if peer.compute_idle:
            pts.add(OperatingPoint.CFK)         # additionally needs peer compute
    pts.add(OperatingPoint.HOST)                # host is always available (slow)
    if deploy.tp_enabled and ctx_tokens <= deploy.tp_capacity_tokens:
        pts.add(OperatingPoint.TP)
    return pts


def _single_capacity_tokens(geom: Geometry, deploy: Deployment) -> int:
    """Single-GPU token capacity, scaled from the measured MHA ceiling by the
    per-token KV ratio (lighter GQA KV -> more tokens fit)."""
    mha = Geometry.llama2_7b_mha()
    base_kv_bytes = deploy.single_capacity_tokens * mha.kv_bytes_per_token
    return int(base_kv_bytes / geom.kv_bytes_per_token)


@dataclass
class Decision:
    point: OperatingPoint
    predicted_ms: float
    admissible: set = field(default_factory=set)
    meets_deadline: bool = True
    reason: str = ""


def select_point(ctx_tokens: int, geom: Geometry, peer: PeerState,
                 deploy: Deployment, model: DecodeStepModel,
                 deadline_ms: "float | None" = None) -> Decision:
    """The deadline-gated cost-model selector: pick the admissible corner with the
    lowest predicted decode-step latency; if none meets the deadline, pick the
    fastest anyway and flag the SLO miss. TP, when admissible but without a
    measured TPOT, is treated as admissibility-only (predict inf) so it never
    silently wins the latency oracle on a throughput-only number."""
    adm = admissible_points(ctx_tokens, geom, peer, deploy)
    if not adm:
        return Decision(OperatingPoint.INFEASIBLE, math.inf, adm, False, "no admissible point")
    preds = {p: model.predict_ms(p, ctx_tokens, peer, deploy) for p in adm}
    best = min(preds, key=preds.get)
    meets = deadline_ms is None or preds[best] <= deadline_ms
    why = []
    if OperatingPoint.SINGLE not in adm:
        why.append("ctx OOMs single GPU")
    if OperatingPoint.CFK not in adm and OperatingPoint.COPYBACK in adm:
        why.append("peer compute busy -> CFK inadmissible, copy-back only")
    if not link_degraded(peer):
        pass
    else:
        why.append("NVLink degraded -> route to host")
    return Decision(best, preds[best], adm, meets, "; ".join(why) or "fastest admissible")


def link_degraded(peer: PeerState) -> bool:
    return peer.nvlink_bw_gbps < BETA_PCIE_GBPS


# --------------------------------------------------------------------------- #
# Do-no-harm hard invariant (spec: collaboration_plan/04_cross_cutting.md SS1) #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LinkState:
    """Live interconnect state at decision time.

    ``transfer_dir`` is a *recorded* field, NOT a constraint: Track A's
    controlled measurement shows push/pull are equivalent in bandwidth and
    victim cost, so direction is not a do-no-harm axis (the old R1-direction
    rule is deliberately absent)."""
    nvlink_eff_gbps: float = BETA_NVLINK_GBPS
    pcie_eff_gbps: float = BETA_PCIE_GBPS
    transfer_dir: str = "push"

    @classmethod
    def from_peer(cls, peer: PeerState) -> "LinkState":
        return cls(nvlink_eff_gbps=peer.nvlink_bw_gbps)


class DoNoHarmViolation(RuntimeError):
    """Raised when a chosen OperatingPoint would break a do-no-harm rule.
    This is a HARD invariant: it is asserted in the hot path and tested in CI
    (tests/test_no_harm.py)."""


def enforce_do_no_harm(point: OperatingPoint,
                       ctx_tokens: int,
                       geom: Geometry,
                       peer: PeerState,
                       link: "LinkState | None" = None,
                       deploy: "Deployment | None" = None,
                       single_capacity_tokens: "int | None" = None) -> OperatingPoint:
    """Single truth source for the R1 do-no-harm rules. Returns ``point``
    unchanged when legal; raises :class:`DoNoHarmViolation` otherwise.

    The selector's exit (runtime.selector.online_select) and the vLLM hot path
    (vllm_integration.peerkv_attn) both route through this function -- the
    invariant lives in exactly one place.

    Field mapping vs the 04_cross_cutting spec: ``peer.compute_idle`` is the
    pre-existing equivalent of the spec's ``compute_busy`` (reused, not
    duplicated); single-GPU capacity comes from ``deploy`` (measured
    serve_m1_tp2 ceiling) or an explicit ``single_capacity_tokens``."""
    if link is None:
        link = LinkState.from_peer(peer)
    cap = single_capacity_tokens
    if cap is None and deploy is not None:
        cap = _single_capacity_tokens(geom, deploy)
    # R1-fit: fits a single GPU => must be SINGLE (never slow down a request
    # that the baseline could serve).
    if cap is not None and ctx_tokens <= cap and point is not OperatingPoint.SINGLE:
        raise DoNoHarmViolation(
            f"R1-fit: fits single ({ctx_tokens} <= {cap} tokens) but chose {point.value}")
    # R1-busy: peer is computing => CFK forbidden (lender loses 33% FLOPs, e23).
    if point is OperatingPoint.CFK and not peer.compute_idle:
        raise DoNoHarmViolation(
            "R1-busy: peer compute-busy, CFK forbidden "
            f"(lender retains only {LENDER_FLOPS_RETAINED:.2%} FLOPs, e23)")
    # R1-route: NVLink effective bandwidth below PCIe => peer-NVLink paths
    # forbidden; spill routes to host instead.
    if (point in (OperatingPoint.CFK, OperatingPoint.COPYBACK)
            and link.nvlink_eff_gbps < link.pcie_eff_gbps):
        raise DoNoHarmViolation(
            f"R1-route: nvlink {link.nvlink_eff_gbps:.1f} < pcie "
            f"{link.pcie_eff_gbps:.1f} GB/s, route host")
    # R1-tp-oracle (R3 red line): TP may be chosen only on a *measured* TPOT,
    # never on the analytic bound or an inf prediction.
    if (point is OperatingPoint.TP and deploy is not None
            and deploy.tp_tpot_ms_per_token is None):
        raise DoNoHarmViolation(
            "R1-tp-oracle: TP chosen without a measured TPOT "
            "(admissibility-only corner, R3)")
    return point

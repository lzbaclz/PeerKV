"""e33 -- offline policy-regret + decision confusion matrix (zero new GPU runs).

The headline of the repositioned paper: a deadline-gated cost-model SELECTOR that
picks the best (phi,kappa,W) corner per request never loses to the best ADMISSIBLE
fixed point, because it picks it. Each existing system is a fixed corner:

    single-only        SINGLE if fits else HOST      vertical offload (FlexGen-like)
    copyback-only      COPYBACK if adm. else HOST    Harvest / AQUA (always borrow HBM)
    cfk-only           CFK if adm. else COPYBACK/HOST DistAttention / Tree (always distribute)
    tp-only            TP if deployable & fits        tensor parallelism

We ingest ONLY committed measured artifacts (real_weights_*, overflow_e2e_*),
calibrate the decode-step model on the FITTING points, hold out the OOM points
(RQ1 error), then run every policy over a parametrized trace. The selector decides
from the cost model (analytic), the oracle from the measured table -- so if the
model mis-ranks, regret > 1 and the confusion matrix shows it. The result is NOT
circular: the model is graded against measurements it was not fit on, and the
copy-back<->CFK boundary is set by the e23 contention measurement, which no single
latency table contains.

Run:  python3 experiments/e33_policy_regret.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umallm.elastic_policy import (  # noqa: E402
    Decision, DecodeStepModel, Deployment, Geometry, OperatingPoint, PeerState,
    admissible_points, select_point,
)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
INF = float("inf")


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def build_measured_table() -> dict:
    """{(geom, ctx): {point: ms_per_token}} from committed JSON artifacts."""
    fit = {
        ("MHA", 16384): "real_weights_L32_C16384.json",
        ("MHA", 32768): "real_weights_L32_C32768.json",
        ("GQA", 16384): "real_weights_L32_C16384_gqa8.json",
        ("GQA", 32768): "real_weights_L32_C32768_gqa8.json",
    }
    oom = {
        ("MHA", 143360): "overflow_e2e_G35.json",
        ("GQA", 573440): "overflow_e2e_G35_gqa8.json",
    }
    table = {}
    for key, fname in fit.items():
        d = _load(fname)
        table[key] = {
            OperatingPoint.SINGLE: d["single_gpu"]["ms_per_token"],
            OperatingPoint.CFK: d["peer_parallel"]["ms_per_token"],
            OperatingPoint.COPYBACK: d["copyback"]["ms_per_token"],
            "_src": fname,
        }
    for key, fname in oom.items():
        d = _load(fname)
        table[key] = {
            OperatingPoint.SINGLE: INF,  # measured OOM
            OperatingPoint.CFK: d["peer_parallel"]["ms_per_token"],
            OperatingPoint.COPYBACK: d["copyback"]["ms_per_token"],
            OperatingPoint.HOST: d["host"]["ms_per_token"],
            "_src": fname,
        }
    return table


def calibrate(table) -> dict:
    """Fit a DecodeStepModel per geometry on the fitting points; return models +
    held-out RQ1 prediction error on the OOM points."""
    geoms = {"MHA": Geometry.llama2_7b_mha(), "GQA": Geometry.gqa_8kv()}
    oom_ctx = {"MHA": 143360, "GQA": 573440}
    models, rq1 = {}, {}
    for g, geom in geoms.items():
        single_pts = {16384: table[(g, 16384)][OperatingPoint.SINGLE],
                      32768: table[(g, 32768)][OperatingPoint.SINGLE]}
        cfk_pt = (16384, table[(g, 16384)][OperatingPoint.CFK])
        cb_pt = (16384, table[(g, 16384)][OperatingPoint.COPYBACK])
        m = DecodeStepModel.calibrate(geom, single_pts, cfk_pt, cb_pt)
        models[g] = m
        # held-out: predict 32K (CFK, copyback) and the OOM point (CFK, copyback)
        peer = PeerState()
        dep = Deployment()
        errs = {}
        for ctx in (32768, oom_ctx[g]):
            for pt in (OperatingPoint.CFK, OperatingPoint.COPYBACK):
                meas = table[(g, ctx)][pt]
                pred = m.predict_ms(pt, ctx, peer, dep)
                errs[f"{pt.value}@{ctx}"] = {
                    "pred_ms": round(pred, 2), "meas_ms": round(meas, 2),
                    "abs_pct": round(100 * abs(pred - meas) / meas, 1),
                }
        rq1[g] = {"A_ms": round(m.A_ms, 2), "beta_eff_gbps": round(m.beta_eff_gbps, 1),
                  "roundtrip_us_per_layer": round(m.roundtrip_ms_per_layer * 1e3, 1),
                  "copyback_eff": round(m.copyback_eff, 3), "heldout": errs}
    return models, rq1


# ---------------------------------------------------------------- workload ----
def make_trace(geom_name, fit_frac=0.7, busy_frac_of_overflow=0.33):
    """A reproducible TP-UNAVAILABLE trace (single-GPU deploy + occasional
    overflow + sometimes-busy peer) -- the regime where the policy matters.
    Contexts restricted to measured points. Returns list of request dicts."""
    fit_ctx = {"MHA": [16384, 32768], "GQA": [16384, 32768]}[geom_name]
    oom_ctx = {"MHA": 143360, "GQA": 573440}[geom_name]
    N = 1000
    n_fit = int(N * fit_frac)
    n_oom = N - n_fit
    n_busy = int(n_oom * busy_frac_of_overflow)
    trace = []
    for i in range(n_fit):
        trace.append({"ctx": fit_ctx[i % len(fit_ctx)], "peer_idle": True})
    for i in range(n_oom):
        trace.append({"ctx": oom_ctx, "peer_idle": i >= n_busy})  # first n_busy busy
    return trace


# ---------------------------------------------------------------- policies ----
def realized_ms(table, geom, ctx, point):
    v = table[(geom, ctx)].get(point, INF)
    return v if v is not None else INF


def oracle_choice(table, geom, ctx, adm):
    cells = {p: realized_ms(table, geom, ctx, p) for p in adm}
    return min(cells, key=cells.get)


def fixed_policy_choice(name, adm):
    """Each existing system pinned to its corner, with an honest fallback chain."""
    order = {
        "single_only": [OperatingPoint.SINGLE, OperatingPoint.HOST],
        "copyback_only": [OperatingPoint.COPYBACK, OperatingPoint.HOST],
        "cfk_only": [OperatingPoint.CFK, OperatingPoint.COPYBACK, OperatingPoint.HOST],
    }[name]
    for p in order:
        if p in adm:
            return p
    return OperatingPoint.HOST


def run(geom_name, models, table, fit_frac=0.7):
    geom = {"MHA": Geometry.llama2_7b_mha(), "GQA": Geometry.gqa_8kv()}[geom_name]
    model = models[geom_name]
    trace = make_trace(geom_name, fit_frac=fit_frac)
    dep = Deployment(tp_enabled=False)  # TP-unavailable regime (the point)
    policies = ["elastic", "oracle", "single_only", "copyback_only", "cfk_only"]
    totals = {p: 0.0 for p in policies}
    confusion = {}  # (selector_point, oracle_point) -> count
    avoided = {"fit_cfk_regret_sum": 0.0, "fit_single_sum": 0.0, "n_fit": 0}
    for req in trace:
        ctx, peer_idle = req["ctx"], req["peer_idle"]
        peer = PeerState(compute_idle=peer_idle)
        adm = admissible_points(ctx, geom, peer, dep)
        orc = oracle_choice(table, geom_name, ctx, adm)
        dec: Decision = select_point(ctx, geom, peer, dep, model)
        sel = dec.point
        totals["oracle"] += realized_ms(table, geom_name, ctx, orc)
        totals["elastic"] += realized_ms(table, geom_name, ctx, sel)
        for fp in ("single_only", "copyback_only", "cfk_only"):
            totals[fp] += realized_ms(table, geom_name, ctx, fixed_policy_choice(fp, adm))
        confusion[(sel.value, orc.value)] = confusion.get((sel.value, orc.value), 0) + 1
        if ctx in ({"MHA": [16384, 32768], "GQA": [16384, 32768]}[geom_name]):
            avoided["n_fit"] += 1
            avoided["fit_cfk_regret_sum"] += realized_ms(table, geom_name, ctx, OperatingPoint.CFK)
            avoided["fit_single_sum"] += realized_ms(table, geom_name, ctx, OperatingPoint.SINGLE)
    base = totals["oracle"]
    regret = {p: round(totals[p] / base, 4) for p in policies}
    return {
        "geom": geom_name, "fit_frac": fit_frac, "n_requests": len(trace),
        "regret_vs_oracle": regret,
        "confusion_selector_vs_oracle": {f"{k[0]}|{k[1]}": v for k, v in confusion.items()},
        "tree_distattn_avoided_distribution_regret_on_fitting": round(
            avoided["fit_cfk_regret_sum"] / avoided["fit_single_sum"], 3),
    }


def tp_available_analysis(table):
    """Fully-MEASURED best-corner table WITH TP in the oracle (closes the gap where
    TP was admissibility-only). TP latency from e37 (tp_tpot.json); single/CFK/
    copyback from e27/e31. Only contexts with both sources. 143K TP uses the 131K
    measured proxy (flagged) since TP-2 fits 253K > 143K."""
    try:
        tp = _load("tp_tpot.json")["contexts"]
    except Exception:
        return None
    tp_ms = {16384: tp["16384"]["tp2"]["ms_per_token"],
             32768: tp["32768"]["tp2"]["ms_per_token"],
             143360: tp["131072"]["tp2"]["ms_per_token"]}  # 131K proxy (flagged)
    rows = []
    for ctx in (16384, 32768, 143360):
        cells = {"single": table[("MHA", ctx)][OperatingPoint.SINGLE],
                 "TP": tp_ms[ctx],
                 "CFK": table[("MHA", ctx)][OperatingPoint.CFK],
                 "copyback": table[("MHA", ctx)][OperatingPoint.COPYBACK]}
        finite = {k: v for k, v in cells.items() if v != INF}
        best = min(finite, key=finite.get)
        rows.append({"ctx": ctx, **{k: (round(v, 2) if v != INF else "OOM") for k, v in cells.items()},
                     "best_corner": best,
                     "tp_proxy": (ctx == 143360)})
    return {"note": ("Fully measured (e37 TP + e27/e31). 143K TP = 131K proxy. Shows TP "
                     "is the best corner where deployable (32K-253K), single wins at 16K "
                     "batch=1 (TP overhead), and beyond 253K TP OOMs -> CFK enablement."),
            "rows": rows}


def main():
    table = build_measured_table()
    models, rq1 = calibrate(table)
    tp_avail = tp_available_analysis(table)
    out = {
        "_experiment": "e33_policy_regret",
        "_is_measured": "derived-from-measured (offline; no new GPU runs)",
        "_sources": sorted({table[k]["_src"] for k in table}),
        "rq1_costmodel_heldout_error": rq1,
        "tp_available_best_corner": tp_avail,
        "runs": [],
        "notes": (
            "Selector decides from the analytic cost model; oracle from the measured "
            "table; held-out OOM points grade the model (RQ1). copy-back<->CFK boundary "
            "is set by the e23 contention measurement (peer-busy => CFK inadmissible), "
            "not derivable from any single latency table -> not circular. Small "
            "measured grid (2 fitting + 1 OOM ctx per geometry): a demonstration, not a "
            "fleet-scale validation."),
    }
    for g in ("MHA", "GQA"):
        for ff in (0.7, 0.5, 0.9):
            out["runs"].append(run(g, models, table, fit_frac=ff))

    path = os.path.join(RESULTS, "policy_regret.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- human summary ----
    print("=" * 78)
    print("RQ1  cost-model held-out prediction error (fit on 16K, predict 32K + OOM)")
    print("=" * 78)
    for g, r in rq1.items():
        print(f"[{g}] A={r['A_ms']}ms beta_eff={r['beta_eff_gbps']}GB/s "
              f"rt={r['roundtrip_us_per_layer']}us/layer cb_eff={r['copyback_eff']}")
        for k, e in r["heldout"].items():
            print(f"      {k:22s} pred {e['pred_ms']:8.2f}  meas {e['meas_ms']:8.2f}"
                  f"  err {e['abs_pct']:5.1f}%")
    print("=" * 78)
    print("RQ3  policy regret vs per-request oracle (1.0 = optimal). TP-unavailable trace.")
    print("=" * 78)
    hdr = f"{'geom':4s} {'fit%':5s} {'elastic':>8s} {'single':>8s} {'copyback':>9s} {'cfk':>8s}  avoided(Tree/DistAttn on fit)"
    print(hdr)
    for r in out["runs"]:
        g = r["regret_vs_oracle"]
        print(f"{r['geom']:4s} {int(r['fit_frac']*100):4d}% "
              f"{g['elastic']:8.3f} {g['single_only']:8.3f} {g['copyback_only']:9.3f} "
              f"{g['cfk_only']:8.3f}  {r['tree_distattn_avoided_distribution_regret_on_fitting']:.2f}x")
    if tp_avail:
        print("=" * 78)
        print("TP-available best corner (MEASURED: e37 TP + e27/e31)  [closes the TP-oracle gap]")
        print("=" * 78)
        print(f"{'ctx':>8s} {'single':>8s} {'TP':>8s} {'CFK':>8s} {'copyback':>9s}  best")
        for r in tp_avail["rows"]:
            tp_s = f"{r['TP']}{'*' if r['tp_proxy'] else ''}"
            print(f"{r['ctx']:8d} {str(r['single']):>8s} {tp_s:>8s} {str(r['CFK']):>8s} "
                  f"{str(r['copyback']):>9s}  -> {r['best_corner']}")
        print("  (* 143K TP = 131K measured proxy; TP-2 fits 253K so 143K is admissible)")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

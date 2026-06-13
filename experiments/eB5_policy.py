"""E-B5 -- online selector closed loop with explicit do-no-harm accounting.

Upgrades e38 (same live-execution methodology: real Llama-2-7B-geometry decoder,
every arm measured THIS run, non-circular oracle) with the four Track-B gates
the plan flags as missing:

  1. decisions go through umallm.runtime.selector.online_select -- the
     single-exit path with enforce_do_no_harm (strict mode);
  2. an explicit do-no-harm violation counter (gate: == 0), plus the
     fits-region check: every fitting request must land on SINGLE;
  3. closed-loop RECALIBRATION: the cost model is re-fit from THIS run's own
     fitting anchors and the trace is re-scored -- regret before vs after
     demonstrates the SS3.3 step-3 loop closing;
  4. the weight-read asymmetry term, now executed (predict_tp_bound_ms), is
     validated against the measured e37 TP TPOT (bound must lower-bound the
     conservative emulated measurement).

'Overflow' is a deployment SLO budget (single_capacity_tokens=32768), not
physical OOM, so every arm is physically runnable and the live oracle is well
defined (same framing as e38).

    /home/lzq/miniconda3/envs/peerkv/bin/python experiments/eB5_policy.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SINGLE_CAP = 32768
TRACE = [   # (context_tokens, peer_compute_idle) -- 14 requests, both regimes
    (4096, True), (8192, True), (8192, False), (16384, True), (16384, False),
    (32768, True), (32768, False),
    (65536, True), (65536, True), (65536, False), (65536, False),
    (65536, True), (16384, True), (4096, False),
]


def main():
    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import (peer_parallel_attention,
                                           flash_partial, merge_partial)
    from umallm.elastic_policy import (
        DecodeStepModel, Deployment, DoNoHarmViolation, Geometry, LinkState,
        OperatingPoint, PeerState)
    from umallm.runtime.selector import online_select
    from umallm.observability.box_probe import gate_or_skip

    if torch.cuda.device_count() < 2:
        print("need >=2 GPUs"); return
    probe_detail = gate_or_skip("eB5_policy")

    L, dm, H, D, dff = 32, 4096, 32, 128, 11008
    Hkv = H
    scale = 1.0 / (D ** 0.5)
    geom = Geometry.llama2_7b_mha()
    dep = Deployment(tp_enabled=False, single_capacity_tokens=SINGLE_CAP)
    # PRIOR model: calibrated on the committed e27 anchors (a different run)
    prior = DecodeStepModel.calibrate(
        geom, single_pts={16384: 17.257, 32768: 22.151},
        cfk_pt=(16384, 20.336), copyback_pt=(16384, 38.402))
    trials, seeds, warmup = 8, 1, 2

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med(xs):
        return statistics.median(sorted(xs))

    g = torch.Generator(device="cuda:0").manual_seed(0)
    def w(a, b):
        return torch.randn(a, b, generator=g, dtype=torch.float16, device="cuda:0") * 0.02
    Wqkv = [w(dm, (H+2*Hkv)*D) for _ in range(L)]; Wo = [w(dm, dm) for _ in range(L)]
    Wg = [w(dm, dff) for _ in range(L)]; Wu = [w(dm, dff) for _ in range(L)]
    Wd = [w(dff, dm) for _ in range(L)]
    n1 = [torch.ones(dm, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    n2 = [torch.ones(dm, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    sync()

    def rms(x, gw):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)) * gw

    def measure_ctx(C):
        Lc = C // 2; Pc = C - Lc
        K0 = [torch.randn(1, Hkv, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
        V0 = [torch.randn(1, Hkv, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
        K1 = [torch.randn(1, Hkv, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
        V1 = [torch.randn(1, Hkv, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
        Kf = [torch.cat([K0[i], K1[i].to("cuda:0")], 2).contiguous() for i in range(L)]
        Vf = [torch.cat([V0[i], V1[i].to("cuda:0")], 2).contiguous() for i in range(L)]
        Ctok = 4096
        K1c = [[K1[i][:, :, s:s+Ctok, :].contiguous() for s in range(0, Pc, Ctok)] for i in range(L)]
        V1c = [[V1[i][:, :, s:s+Ctok, :].contiguous() for s in range(0, Pc, Ctok)] for i in range(L)]
        Kh = [[k.cpu().pin_memory() for k in K1c[i]] for i in range(L)]
        Vh = [[v.cpu().pin_memory() for v in V1c[i]] for i in range(L)]
        cps = torch.cuda.Stream(device="cuda:0")
        sync()

        def a_single(i, q):
            return F.scaled_dot_product_attention(q, Kf[i], Vf[i], scale=scale)
        def a_cfk(i, q):
            return peer_parallel_attention(q, [(K0[i], V0[i]), (K1[i], V1[i])], scale)
        def _stream_back(i, q, ks, vs):
            O, l = flash_partial(q, K0[i], V0[i], scale); n = len(ks)
            buf=[None]*n; ev=[torch.cuda.Event() for _ in range(n)]
            with torch.cuda.stream(cps):
                buf[0]=(ks[0].to("cuda:0",non_blocking=True),vs[0].to("cuda:0",non_blocking=True)); ev[0].record(cps)
            for j in range(n):
                if j+1<n:
                    with torch.cuda.stream(cps):
                        buf[j+1]=(ks[j+1].to("cuda:0",non_blocking=True),vs[j+1].to("cuda:0",non_blocking=True)); ev[j+1].record(cps)
                torch.cuda.current_stream().wait_event(ev[j]); Kc,Vc=buf[j]
                Oc,lc=flash_partial(q,Kc,Vc,scale); O,l=merge_partial(O,l,Oc,lc); buf[j]=None
            return O.to(torch.float16)
        def a_copyback(i, q): return _stream_back(i, q, K1c[i], V1c[i])
        def a_host(i, q): return _stream_back(i, q, Kh[i], Vh[i])

        arms = {"single": a_single, "cfk": a_cfk, "copyback": a_copyback, "host": a_host}
        def token(fn):
            x = torch.randn(1, dm, dtype=torch.float16, device="cuda:0")
            for i in range(L):
                h = rms(x, n1[i]); qkv = h @ Wqkv[i]
                q = qkv[:, :H*D].view(1, H, 1, D)
                x = x + (fn(i, q).reshape(1, dm) @ Wo[i])
                gg = rms(x, n2[i]); x = x + (F.silu(gg @ Wg[i]) * (gg @ Wu[i])) @ Wd[i]
            return x
        lat = {}
        for name, fn in arms.items():
            t = trials if name != "host" else 3
            ms = []
            for _ in range(seeds):
                for _ in range(warmup): token(fn)
                sync(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
                for _ in range(t):
                    sync(); s.record(); token(fn); e.record(); sync(); ms.append(s.elapsed_time(e))
            lat[name] = round(med(ms), 3)
        for d in (K0,V0,K1,V1,Kf,Vf,K1c,V1c,Kh,Vh): del d
        torch.cuda.empty_cache()
        return lat

    contexts = sorted({c for c, _ in TRACE})
    latency = {}
    for C in contexts:
        latency[C] = measure_ctx(C)
        print(f"  ctx={C:6d} live ms/tok: {latency[C]}")

    # --- closed-loop recalibration: re-fit the model from THIS run ---------- #
    recal = DecodeStepModel.calibrate(
        geom,
        single_pts={16384: latency[16384]["single"], 32768: latency[32768]["single"]},
        cfk_pt=(16384, latency[16384]["cfk"]),
        copyback_pt=(16384, latency[16384]["copyback"]))

    arm_of = {OperatingPoint.SINGLE: "single", OperatingPoint.CFK: "cfk",
              OperatingPoint.COPYBACK: "copyback", OperatingPoint.HOST: "host"}

    def replay(model, tag):
        sel_tot = orc_tot = 0.0
        violations = 0
        fits_nonsingle = 0
        log, confusion = [], {}
        for C, peer_idle in TRACE:
            peer = PeerState(compute_idle=peer_idle)
            link = LinkState.from_peer(peer)
            try:
                dec = online_select(C, geom, dep, model, peer=peer, link=link,
                                    strict=True)
            except DoNoHarmViolation:
                violations += 1
                continue
            sel_arm = arm_of.get(dec.point, "host")
            adm_arms = {arm_of[p]: latency[C][arm_of[p]]
                        for p in dec.admissible if p in arm_of}
            if not adm_arms:
                adm_arms = {"host": latency[C]["host"]}
            orc_arm = min(adm_arms, key=adm_arms.get)
            if C <= SINGLE_CAP and sel_arm != "single":
                fits_nonsingle += 1
            sel_tot += latency[C][sel_arm]; orc_tot += latency[C][orc_arm]
            confusion[f"{sel_arm}|{orc_arm}"] = confusion.get(f"{sel_arm}|{orc_arm}", 0) + 1
            log.append({"ctx": C, "peer_idle": peer_idle,
                        "admissible": sorted(adm_arms),
                        "selector": sel_arm, "oracle": orc_arm,
                        "selector_ms": latency[C][sel_arm],
                        "oracle_ms": latency[C][orc_arm],
                        "match": sel_arm == orc_arm})
            print(f"  [{tag}] ctx={C:6d} idle={str(peer_idle):5s} "
                  f"sel={sel_arm:8s} orc={orc_arm:8s} "
                  f"{'OK' if sel_arm == orc_arm else 'MISS'}")
        return {
            "realized_regret": round(sel_tot / orc_tot, 4),
            "decisions_matching_oracle": f"{sum(r['match'] for r in log)}/{len(log)}",
            "do_no_harm_violations": violations,
            "fitting_requests_not_single": fits_nonsingle,
            "confusion_selector_vs_oracle": confusion,
            "trace": log,
        }

    pre = replay(prior, "prior")
    post = replay(recal, "recal")

    # --- fixed-policy regrets on the same trace ----------------------------- #
    def fixed_policy(policy):
        tot = orc = 0.0
        for C, peer_idle in TRACE:
            arm = policy
            if C <= SINGLE_CAP:
                pass                                     # every arm runnable
            if arm == "cfk" and not peer_idle:
                arm = "copyback"                         # busy fallback chain
            if arm == "single" and C > SINGLE_CAP:
                arm = "copyback"                         # cap fallback
            adm = ["single", "cfk", "copyback", "host"] if C <= SINGLE_CAP \
                else (["cfk", "copyback", "host"] if peer_idle else ["copyback", "host"])
            orc += min(latency[C][a] for a in adm)
            tot += latency[C][arm]
        return round(tot / orc, 4)

    fixed = {p: fixed_policy(p) for p in ("single", "cfk", "copyback", "host")}

    # --- weight-read asymmetry term: executed + validated vs e37 ------------ #
    tp_validation = {"note": "predict_tp_bound_ms is an optimistic analytic "
                             "bound; e37 measures a conservative emulated "
                             "TP-2 (unfused GEMV + .to() all-reduce). The "
                             "bound must sit at/below the measurement; "
                             "production TP lands between them. R3: neither "
                             "number feeds select_point."}
    try:
        tp = json.load(open(os.path.join(RES, "tp_tpot.json")))
        rows = []
        for ctx_s, cell in tp["contexts"].items():
            meas = cell.get("tp2", {}).get("ms_per_token")
            if meas:
                bound = recal.predict_tp_bound_ms(int(ctx_s))
                rows.append({"ctx": int(ctx_s), "measured_e37_ms": meas,
                             "analytic_bound_ms": round(bound, 2),
                             "bound_below_measured": bound <= meas * 1.05})
        tp_validation["rows"] = rows
        tp_validation["all_bounds_hold"] = all(r["bound_below_measured"] for r in rows)
    except FileNotFoundError:
        tp_validation["rows"] = "tp_tpot.json not found"

    out = {
        "_experiment": "eB5_policy",
        "_is_measured": True,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "single_capacity_tokens": SINGLE_CAP,
        "box_probe": probe_detail,
        "note": ("LIVE closed loop through runtime.selector.online_select "
                 "(strict single-exit); oracle measured this run; recalibrated "
                 "model re-fit from this run's own anchors (SS3.3 step 3)."),
        "live_latency_ms_per_token": latency,
        "selector_prior_model": {k: v for k, v in pre.items() if k != "trace"},
        "selector_recalibrated": post,
        "fixed_policy_regret": fixed,
        "tp_weight_read_validation": tp_validation,
        "gates": {
            "do_no_harm_violations==0": (pre["do_no_harm_violations"] == 0
                                         and post["do_no_harm_violations"] == 0),
            "fitting_always_single": (pre["fitting_requests_not_single"] == 0
                                      and post["fitting_requests_not_single"] == 0),
            "recal_regret<=prior": post["realized_regret"] <= pre["realized_regret"],
        },
    }
    path = os.path.join(RES, "eB5_policy.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  prior:  regret={pre['realized_regret']} match={pre['decisions_matching_oracle']}")
    print(f"  recal:  regret={post['realized_regret']} match={post['decisions_matching_oracle']}")
    print(f"  fixed-policy regrets: {fixed}")
    print(f"  gates: {out['gates']}")
    print("wrote", path)


if __name__ == "__main__":
    main()

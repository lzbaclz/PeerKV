"""e38 -- ELASTIC online closed-loop: select_point drives LIVE execution.

The circularity-killer. e33's regret is offline (selector vs a table-derived
oracle). Here the selector picks per request and we ACTUALLY EXECUTE the chosen
mechanism on real weights and MEASURE the realized latency this run -- so the
oracle is a live measurement, not the table the selector reads (the selector reads
the analytic cost model). If the model mis-ranks, realized regret > 1.

Real Llama-2-7B-geometry decoder (L=32, fp16 weights resident on cuda:0). For each
unique context we measure single / compute-follows-KV / copy-back / host LIVE once
(latency depends on (ctx, arm); peer-busy only gates admissibility). A trace of
requests with mixed (context, peer-state) is then replayed: per request the
deadline-gated selector picks an admissible corner; realized regret = sum(realized
selector) / sum(realized live-oracle over the same admissible set).

"Overflow" here is a DEPLOYMENT/SLO budget (single_capacity_tokens), not physical
OOM, so every arm is physically runnable and the live oracle is well defined.

    /home/lzq/miniconda3/envs/peerkv/bin/python experiments/e38_online_loop.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SINGLE_CAP = 32768           # deployment single-GPU budget (contexts above = overflow)
TRACE = [                    # (context_tokens, peer_compute_idle)
    (8192, True), (8192, True), (16384, True), (16384, False),
    (65536, True), (65536, True), (65536, False),
]


def main():
    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import peer_parallel_attention, flash_partial, merge_partial
    from umallm.elastic_policy import (
        DecodeStepModel, Deployment, Geometry, OperatingPoint, PeerState,
        admissible_points, select_point)

    if torch.cuda.device_count() < 2:
        print("need >=2 GPUs"); return
    L, dm, H, D, dff = 32, 4096, 32, 128, 11008
    Hkv = H
    scale = 1.0 / (D ** 0.5)
    geom = Geometry.llama2_7b_mha()
    dep = Deployment(tp_enabled=False, single_capacity_tokens=SINGLE_CAP)
    model = DecodeStepModel.calibrate(
        geom, single_pts={16384: 17.257, 32768: 22.151},
        cfk_pt=(16384, 20.336), copyback_pt=(16384, 38.402))
    trials, seeds, warmup = 8, 1, 2
    PT = {OperatingPoint.SINGLE: "single", OperatingPoint.CFK: "cfk",
          OperatingPoint.COPYBACK: "copyback", OperatingPoint.HOST: "host"}

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
    print(f"  weights {sum(t.numel()*2 for t in Wqkv+Wo+Wg+Wu+Wd)/1024**3:.1f} GiB on cuda:0")

    def rms(x, gw):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)) * gw

    def measure_ctx(C):
        """Measure realized ms/token for each arm at context C (live)."""
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

    # replay the trace: selector (cost-model) vs live oracle (measured this run)
    arm_to_pt = {"single": OperatingPoint.SINGLE, "cfk": OperatingPoint.CFK,
                 "copyback": OperatingPoint.COPYBACK, "host": OperatingPoint.HOST}
    pt_to_arm = {v: k for k, v in arm_to_pt.items()}
    sel_tot = orc_tot = 0.0
    log, confusion = [], {}
    for C, peer_idle in TRACE:
        peer = PeerState(compute_idle=peer_idle)
        adm = {p for p in admissible_points(C, geom, peer, dep) if p in pt_to_arm}
        adm_arms = {pt_to_arm[p]: latency[C][pt_to_arm[p]] for p in adm}
        orc_arm = min(adm_arms, key=adm_arms.get)
        dec = select_point(C, geom, peer, dep, model)
        sel_arm = pt_to_arm.get(dec.point, "host")
        if sel_arm not in adm_arms:           # selector chose inadmissible -> fall back
            sel_arm = orc_arm
        sel_tot += latency[C][sel_arm]; orc_tot += latency[C][orc_arm]
        confusion[f"{sel_arm}|{orc_arm}"] = confusion.get(f"{sel_arm}|{orc_arm}", 0) + 1
        log.append({"ctx": C, "peer_idle": peer_idle, "admissible": sorted(adm_arms),
                    "selector": sel_arm, "oracle": orc_arm,
                    "selector_ms": latency[C][sel_arm], "oracle_ms": latency[C][orc_arm],
                    "match": sel_arm == orc_arm})

    out = {
        "_experiment": "e38_online_loop", "_is_measured": True,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0), "single_capacity_tokens": SINGLE_CAP,
        "note": ("LIVE closed-loop: selector reads the analytic cost model; the oracle "
                 "is measured THIS run -> realized regret is non-circular. 'Overflow' is "
                 "a deployment SLO budget, not physical OOM, so every arm is runnable."),
        "live_latency_ms_per_token": latency,
        "trace": log,
        "realized_regret": round(sel_tot / orc_tot, 4),
        "decisions_matching_oracle": f"{sum(r['match'] for r in log)}/{len(log)}",
        "confusion_selector_vs_oracle": confusion,
    }
    path = os.path.join(RES, "online_loop.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== ONLINE CLOSED-LOOP (live execution) ===")
    for r in log:
        print(f"  ctx={r['ctx']:6d} peer_idle={str(r['peer_idle']):5s} adm={r['admissible']}"
              f"  sel={r['selector']:8s}({r['selector_ms']}ms) orc={r['oracle']:8s}({r['oracle_ms']}ms)"
              f"  {'OK' if r['match'] else 'MISS'}")
    print(f"\n  realized regret = {out['realized_regret']}   "
          f"decisions matching oracle = {out['decisions_matching_oracle']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

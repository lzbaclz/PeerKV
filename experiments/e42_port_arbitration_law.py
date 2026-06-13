"""e42 -- the HBM-read-port arbitration law (the paper's characterization spine).

Overlap-safe timing: sustained holder load on holder_stream + CUDA events on
copy_stream only (see overlap_safe_bw.py). Never torch.cuda.synchronize(holder_device)
during copy measurement.
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
from datetime import datetime, timezone
from pathlib import Path

from overlap_safe_bw import (
    drain_stream,
    measure_bw_events,
    start_sustained_load,
)

OUT = Path(__file__).resolve().parent / "results" / "port_arbitration_law.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import torch

    def bufs(mb):
        n = mb * 1024 * 1024 // 2
        return n, n * 2

    def mk_gemm(nn):
        A = torch.randn(nn, nn, dtype=torch.float16, device="cuda:1")
        B = torch.randn(nn, nn, dtype=torch.float16, device="cuda:1")
        return (lambda: torch.mm(A, B)), 2 * nn**3, 3 * nn * nn * 2

    def mk_skinny(m, k, r):
        A = torch.randn(m, k, dtype=torch.float16, device="cuda:1")
        B = torch.randn(k, r, dtype=torch.float16, device="cuda:1")
        return (lambda: torch.mm(A, B)), 2 * m * k * r, (m * k + k * r + m * r) * 2

    op_zoo = []
    for nn in (1024, 2048, 4096, 8192):
        op, fl, by = mk_gemm(nn)
        op_zoo.append((f"gemm{nn}", op, fl, by, fl / by))
    for (m, k, r, tag) in [(16384, 16384, 8, "decode_gemv8"), (16384, 16384, 64, "decode_b64"),
                           (4096, 11008, 16, "ffn_b16")]:
        op, fl, by = mk_skinny(m, k, r)
        op_zoo.append((tag, op, fl, by, fl / by))

    membound_op = dict((t, (op, fl, by, ai)) for t, op, fl, by, ai in op_zoo)["decode_gemv8"]
    mem_op = membound_op[0]
    holder_ls = torch.cuda.Stream(device=1)
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)

    n, nbytes = bufs(256)
    src1 = torch.ones(n, dtype=torch.float16, device="cuda:1")
    dst0 = torch.empty(n, dtype=torch.float16, device="cuda:0")

    def pull():
        with torch.cuda.stream(s0):
            dst0.copy_(src1, non_blocking=True)

    def push():
        with torch.cuda.stream(s1):
            dst0.copy_(src1, non_blocking=True)

    def busy_pull():
        start_sustained_load(mem_op, holder_ls)
        return measure_bw_events(
            pull, s0, nbytes, args.trials,
            holder_stream=holder_ls, holder_op=mem_op, assert_holder_busy=True,
        )

    def busy_push():
        start_sustained_load(mem_op, holder_ls)
        return measure_bw_events(
            push, s1, nbytes, args.trials,
            holder_stream=holder_ls, holder_op=mem_op, assert_holder_busy=True,
        )

    # ---------- (1) push vs pull, idle vs memory-bound holder ----------
    push_pull = []
    for _ in range(args.seeds):
        pi, _ = measure_bw_events(pull, s0, nbytes, args.trials)
        pui, _ = measure_bw_events(push, s1, nbytes, args.trials)
        pm, _ = busy_pull()
        drain_stream(holder_ls)
        pum, _ = busy_push()
        drain_stream(holder_ls)
        push_pull.append({
            "pull_idle": pi, "push_idle": pui, "pull_busy": pm, "push_busy": pum,
            "pull_retained": pm / pi, "push_retained": pum / pui,
            "push_adv_idle_pp": (pui / pi - 1) * 100,
            "push_adv_busy_pp": (pum / pm - 1) * 100,
        })

    def agg(key):
        return [r[key] for r in push_pull]

    pp = {k: round(statistics.mean(agg(k)), 3) for k in push_pull[0]}
    pp["push_adv_busy_pp_std"] = round(statistics.pstdev(agg("push_adv_busy_pp")), 2) if args.seeds > 1 else 0
    pp["pull_busy_std"] = round(statistics.pstdev(agg("pull_busy")), 2) if args.seeds > 1 else 0
    pp["push_busy_std"] = round(statistics.pstdev(agg("push_busy")), 2) if args.seeds > 1 else 0
    pp["_timing"] = "overlap_safe_events"

    # ---------- (2) 4-cell R/W isolation ----------
    a1 = torch.ones(n, dtype=torch.float16, device="cuda:1")
    b1 = torch.empty(n, dtype=torch.float16, device="cuda:1")
    a0 = torch.ones(n, dtype=torch.float16, device="cuda:0")
    b0 = torch.empty(n, dtype=torch.float16, device="cuda:0")
    cells = {
        "R_remote": (lambda: b0.copy_(a1, non_blocking=True), s0),
        "W_remote": (lambda: b1.copy_(a0, non_blocking=True), s0),
        "R_local": (lambda: b0.copy_(a1, non_blocking=True), s1),
        "W_local": (lambda: b1.copy_(a0, non_blocking=True), s1),
    }

    rw_retained_seeds = {k: [] for k in cells}
    rw_idle_seeds = {k: [] for k in cells}
    rw_busy_seeds = {k: [] for k in cells}
    for _ in range(args.seeds):
        idle_seed, busy_seed = {}, {}
        for name, (fn, cstream) in cells.items():
            def issue(f=fn, cs=cstream):
                with torch.cuda.stream(cs):
                    f()
            g, _ = measure_bw_events(issue, cstream, nbytes, args.trials)
            idle_seed[name] = g
        start_sustained_load(mem_op, holder_ls)
        for name, (fn, cstream) in cells.items():
            def issue(f=fn, cs=cstream):
                with torch.cuda.stream(cs):
                    f()
            g, _ = measure_bw_events(
                issue, cstream, nbytes, args.trials,
                holder_stream=holder_ls, holder_op=mem_op, assert_holder_busy=True,
            )
            busy_seed[name] = g
        drain_stream(holder_ls)
        for k in cells:
            rw_idle_seeds[k].append(idle_seed[k])
            rw_busy_seeds[k].append(busy_seed[k])
            rw_retained_seeds[k].append(busy_seed[k] / idle_seed[k])

    rw_retained = {k: round(statistics.mean(v), 3) for k, v in rw_retained_seeds.items()}
    rw_isolation_std = {
        k: round(statistics.pstdev(v), 3) if args.seeds > 1 else 0.0
        for k, v in rw_retained_seeds.items()
    }
    rw_idle = {k: round(statistics.mean(v), 1) for k, v in rw_idle_seeds.items()}
    rw_busy = {k: round(statistics.mean(v), 1) for k, v in rw_busy_seeds.items()}

    # ---------- (3) AI sweep ----------
    ai_sweep = []
    for tag, op, fl, by, ai in op_zoo:
        pidle, _ = measure_bw_events(pull, s0, nbytes, args.trials)
        start_sustained_load(op, holder_ls)
        pbusy, _ = measure_bw_events(
            pull, s0, nbytes, args.trials,
            holder_stream=holder_ls, holder_op=op, assert_holder_busy=True,
        )
        drain_stream(holder_ls)
        ai_sweep.append({"op": tag, "AI": round(ai, 1), "pull_retained": round(pbusy / pidle, 3)})

    # ---------- (4) size sweep ----------
    size_sweep = []
    for mb in (4, 16, 64, 128, 256, 512):
        nn, nb2 = bufs(mb)
        s = torch.ones(nn, dtype=torch.float16, device="cuda:1")
        d = torch.empty(nn, dtype=torch.float16, device="cuda:0")

        def pl():
            with torch.cuda.stream(s0):
                d.copy_(s, non_blocking=True)

        def ph():
            with torch.cuda.stream(s1):
                d.copy_(s, non_blocking=True)

        pi, _ = measure_bw_events(pl, s0, nb2, 40)
        pui, _ = measure_bw_events(ph, s1, nb2, 40)
        start_sustained_load(mem_op, holder_ls)
        pm, _ = measure_bw_events(
            pl, s0, nb2, 40, holder_stream=holder_ls, holder_op=mem_op, assert_holder_busy=True,
        )
        pum, _ = measure_bw_events(
            ph, s1, nb2, 40, holder_stream=holder_ls, holder_op=mem_op, assert_holder_busy=True,
        )
        drain_stream(holder_ls)
        size_sweep.append({
            "MB": mb,
            "pull_retained": round(pm / pi, 3),
            "push_retained": round(pum / pui, 3),
            "push_adv_pp": round((pum / pm - 1) * 100, 1),
        })
        del s, d

    # ---------- (5) one-sidedness ----------
    op, fl, by, ai = membound_op
    torch.cuda.set_device(1)
    gs = torch.cuda.Stream(device=1)

    def run_gemm(it):
        with torch.cuda.stream(gs):
            for _ in range(it):
                op()

    run_gemm(10)
    torch.cuda.synchronize(1)
    t0 = time.perf_counter()
    run_gemm(400)
    torch.cuda.synchronize(1)
    alone = fl / ((time.perf_counter() - t0) / 400) / 1e12
    torch.cuda.set_device(0)
    pstream = torch.cuda.Stream(device=0)
    with torch.cuda.stream(pstream):
        for _ in range(4000):
            dst0.copy_(src1, non_blocking=True)
    torch.cuda.set_device(1)
    run_gemm(10)
    torch.cuda.synchronize(1)
    t0 = time.perf_counter()
    run_gemm(400)
    torch.cuda.synchronize(1)
    withpull = fl / ((time.perf_counter() - t0) / 400) / 1e12
    torch.cuda.set_device(0)
    torch.cuda.synchronize(0)
    one_sided = {
        "lender_tflops_alone": round(alone, 1),
        "lender_tflops_with_remote_pull": round(withpull, 1),
        "lender_retained": round(withpull / alone, 3),
    }

    res = {
        "_experiment": "e42_port_arbitration_law",
        "_is_measured": True,
        "_timing_method": "overlap_safe_cuda_events",
        "device": torch.cuda.get_device_name(0),
        "trials": args.trials,
        "seeds": args.seeds,
        "push_vs_pull": pp,
        "rw_isolation_retained": rw_retained,
        "rw_isolation_std": rw_isolation_std,
        "rw_idle_gbs": rw_idle,
        "rw_busy_gbs": rw_busy,
        "ai_sweep": ai_sweep,
        "size_sweep": size_sweep,
        "one_sidedness": one_sided,
        "law": (
            "Overlap-safe measurement: sustained holder load + copy-stream events only. "
            "See push_vs_pull and rw_isolation_retained for read/write asymmetry under "
            "concurrent memory-bound holder."
        ),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"(1) push/pull: idle +{pp['push_adv_idle_pp']:.1f}pp  busy +{pp['push_adv_busy_pp']:.1f}pp "
          f"(pull_ret {pp['pull_retained']:.2f}, push_ret {pp['push_retained']:.2f})")
    print(f"(2) 4-cell retained: {rw_retained}")
    print(f"(3) AI sweep: " + ", ".join(f"{r['op']}(AI{r['AI']:.0f}):{r['pull_retained']:.2f}" for r in ai_sweep))
    print(f"(4) size sweep push_adv_pp: " + ", ".join(f"{r['MB']}MB:{r['push_adv_pp']:.0f}" for r in size_sweep))
    print(f"(5) one-sided lender retained: {one_sided['lender_retained']}")
    print(f"-> wrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"OVERLAP ASSERTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)

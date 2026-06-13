"""e19 -- one-step-ahead PREFETCH (double-buffered) decode step: the paper's
claimed mechanism, measured, with a FAIR overlapped baseline on BOTH tiers.

implementation.tex claims "the prefetch is issued one step ahead ... hiding
NVLink latency behind compute." e17/e18 are the NO-prefetch upper bound and the
host baseline there is synchronous -- the biggest reviewer attack ("your win
evaporates against a pipelined/FlexGen host baseline"). This script gives every
placement the SAME double-buffered prefetch: while attention computes over chunk
i, chunk i+1 is copied on a separate CUDA stream, so the steady-state per-step
cost is max(compute, transfer) per chunk instead of compute+transfer.

Why NVLink should WIN MORE (not less) with prefetch: NVLink's per-chunk transfer
is ~11x smaller, so it hides fully behind compute (step becomes compute-bound);
host's transfer is too large to hide (step stays transfer-bound). So a fair
overlapped comparison moves the ratio UP toward the raw bandwidth ratio, not
down. Peak stays bounded to q + local + 2 chunks (prefetch depth 1).

CUDA-event timed; reports median + IQR over repeats x seeds.
"""
from __future__ import annotations
import argparse, json, statistics
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "p2p_prefetch.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=2048)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--local-blocks", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunk-blocks", type=int, default=32)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    import torch

    H, T, D = args.heads, args.block_tokens, args.head_dim
    N, L = args.n_blocks, min(args.local_blocks, args.n_blocks)
    n_spill = N - L
    C = args.chunk_blocks
    scale = 1.0 / (D ** 0.5)
    dev = torch.device("cuda:0")
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))

    q = torch.randn(1, H, 1, D, dtype=torch.float16, device=dev)
    Kloc = torch.randn(1, H, L * T, D, dtype=torch.float16, device=dev)
    Vloc = torch.randn(1, H, L * T, D, dtype=torch.float16, device=dev)

    def make_chunks(spill_dev):
        d = "cpu" if spill_dev == "host" else spill_dev
        ck, cv, rem = [], [], n_spill
        while rem > 0:
            c = min(C, rem)
            K = torch.randn(1, H, c * T, D, dtype=torch.float16, device=d)
            V = torch.randn(1, H, c * T, D, dtype=torch.float16, device=d)
            if spill_dev == "host":
                K, V = K.pin_memory(), V.pin_memory()
            ck.append(K); cv.append(V); rem -= c
        return ck, cv

    def merge(state, Kc, Vc):
        m, l, o = state
        s = (q * scale @ Kc.transpose(-1, -2)).float()
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        corr = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        o = o * corr + p.to(Vc.dtype) @ Vc
        return (m_new, l, o)

    def fresh_state():
        return (torch.full((1, H, 1, 1), float("-inf"), device=dev, dtype=torch.float32),
                torch.zeros((1, H, 1, 1), device=dev, dtype=torch.float32),
                torch.zeros((1, H, 1, D), device=dev, dtype=torch.float32))

    copy_stream = torch.cuda.Stream(device=dev)

    def step_no_prefetch(ck, cv):
        st = fresh_state()
        st = merge(st, Kloc, Vloc)
        for Kc, Vc in zip(ck, cv):
            st = merge(st, Kc.to(dev, non_blocking=True), Vc.to(dev, non_blocking=True))
        m, l, o = st
        return (o / l).to(q.dtype)

    def step_prefetch(ck, cv):
        # double-buffered one-step-ahead prefetch: copy chunk i+1 on copy_stream
        # while compute merges chunk i on the default stream.
        st = fresh_state()
        st = merge(st, Kloc, Vloc)
        n = len(ck)
        if n == 0:
            m, l, o = st
            return (o / l).to(q.dtype)
        ev = [torch.cuda.Event() for _ in range(n)]
        buf = [None] * n
        with torch.cuda.stream(copy_stream):
            buf[0] = (ck[0].to(dev, non_blocking=True), cv[0].to(dev, non_blocking=True))
            ev[0].record(copy_stream)
        for i in range(n):
            if i + 1 < n:
                with torch.cuda.stream(copy_stream):
                    buf[i + 1] = (ck[i + 1].to(dev, non_blocking=True),
                                  cv[i + 1].to(dev, non_blocking=True))
                    ev[i + 1].record(copy_stream)
            torch.cuda.current_stream().wait_event(ev[i])   # compute waits only for chunk i
            Kc, Vc = buf[i]
            st = merge(st, Kc, Vc)
            buf[i] = None
        m, l, o = st
        return (o / l).to(q.dtype)

    def timed(fn, ck, cv):
        for _ in range(3):
            fn(ck, cv)
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(); ev0.record()
            fn(ck, cv); ev1.record(); torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1))   # ms
        return ts

    def bench(spill_dev):
        allts = {"no_prefetch": [], "prefetch": [], "compute_only": []}
        peak = None
        for sd in range(args.seeds):
            torch.manual_seed(1000 + sd)
            ck, cv = make_chunks(spill_dev)
            # pre-staged RESIDENT copies on cuda:0: step_no_prefetch over these
            # has .to(dev) as a no-op, so it times pure compute (transfer-free).
            ck_res = [k.to(dev) for k in ck]; cv_res = [v.to(dev) for v in cv]
            torch.cuda.reset_peak_memory_stats(0)
            allts["no_prefetch"] += timed(step_no_prefetch, ck, cv)
            allts["prefetch"] += timed(step_prefetch, ck, cv)
            peak = torch.cuda.max_memory_allocated(0) / 1024**2
            allts["compute_only"] += timed(step_no_prefetch, ck_res, cv_res)
            del ck, cv, ck_res, cv_res; torch.cuda.empty_cache()
        def stat(xs):
            xs = sorted(xs); n = len(xs)
            med = statistics.median(xs)
            iqr = xs[int(0.75*n)] - xs[int(0.25*n)] if n >= 4 else 0.0
            return {"median_ms": med, "iqr_ms": iqr, "n": n}
        return {k: stat(v) for k, v in allts.items()}, peak

    nv, nv_peak = bench("cuda:1")
    ho, ho_peak = bench("host")

    def ratio(a, b):
        return b["median_ms"] / a["median_ms"] if a["median_ms"] else None

    res = {
        "_experiment": "e19_prefetch", "_is_measured": True,
        "kind": "double_buffered_prefetch_decode_step",
        "device": torch.cuda.get_device_name(0), "peer_access_enabled": p2p,
        "n_blocks": N, "local_blocks": L, "spill_blocks": n_spill,
        "block_tokens": T, "heads": H, "head_dim": D, "chunk_blocks": C,
        "trials": args.trials, "seeds": args.seeds,
        "nvlink": nv, "host": ho,
        "nvlink_peak_mb": nv_peak, "host_peak_mb": ho_peak,
        "nvlink_vs_host_no_prefetch": ratio(nv["no_prefetch"], ho["no_prefetch"]),
        "nvlink_vs_host_prefetch": ratio(nv["prefetch"], ho["prefetch"]),
        "nvlink_prefetch_hidden_frac": (1.0 - (nv["prefetch"]["median_ms"] - nv["compute_only"]["median_ms"])
                                        / max(1e-9, nv["no_prefetch"]["median_ms"] - nv["compute_only"]["median_ms"])),
        "host_prefetch_hidden_frac": (1.0 - (ho["prefetch"]["median_ms"] - ho["compute_only"]["median_ms"])
                                      / max(1e-9, ho["no_prefetch"]["median_ms"] - ho["compute_only"]["median_ms"])),
        "note": ("FAIR overlapped (FlexGen-style) baseline on BOTH tiers: one-step-ahead "
                 "double-buffered prefetch on a side CUDA stream. nvlink_vs_host_prefetch "
                 "is the honest overlapped comparison. hidden_frac = fraction of transfer "
                 "latency hidden behind compute (1.0 = fully hidden -> compute-bound)."),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  spill={n_spill} blk, chunk={C}")
    for tier, d in (("NVLink", nv), ("host", ho)):
        print(f"  {tier:6s}: no_prefetch={d['no_prefetch']['median_ms']:7.2f}  "
              f"prefetch={d['prefetch']['median_ms']:7.2f}  compute_only={d['compute_only']['median_ms']:7.2f} ms")
    print(f"  NVLink-vs-host: no_prefetch={res['nvlink_vs_host_no_prefetch']:.2f}x  "
          f"prefetch={res['nvlink_vs_host_prefetch']:.2f}x")
    print(f"  hidden frac: NVLink={res['nvlink_prefetch_hidden_frac']:.2f}  host={res['host_prefetch_hidden_frac']:.2f}")
    print(f"  peak: NVLink={nv_peak:.0f}MB host={ho_peak:.0f}MB  -> wrote {OUT}")


if __name__ == "__main__":
    main()

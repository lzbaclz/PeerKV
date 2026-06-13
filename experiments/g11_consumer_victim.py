"""G11 -- consumer-side victim: decode on the DESTINATION GPU while the 512MB
handoff lands in its HBM.

G1-G3 measured the *holder* (source) side: its HBM is read by the copy. In the
prefill->decode handoff that motivates disaggregated serving, the latency-critical
GPU is often the *consumer* (destination), whose HBM absorbs the copy's WRITES --
and g10 shows write traffic costs more than read at matched footprint. G11 closes
that gap: decode runs on GPU0 (consumer) while a 512MB handoff arrives GPU1->GPU0:

  peer_in_push  GPU1->GPU0 over NVLink (holder-issued)    ~280 GB/s HBM write on victim
  peer_in_pull  GPU1->GPU0 over NVLink (consumer-issued)  ~280 GB/s HBM write on victim
  host_in       pinned host->GPU0 over PCIe (consumer-issued) ~26 GB/s HBM write

Protocol identical to g3: per-iteration CUDA events on the consumer's decode
stream only, clocks locked, single handoff per seed, no whole-device sync.
Reports the g10 write-curve interpolation at the achieved footprint so the
measured consumer cost can be checked against the paced-injection prediction.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g11_consumer_victim.json"
G10 = RESULTS / "g10_paced_sweep.json"


def g10_write_prediction(frac: float):
    """Interpolate the g10 write-only victim curve at footprint fraction `frac`."""
    if not G10.exists():
        return None
    g10 = json.loads(G10.read_text())
    pts = sorted(
        (v["achieved_frac"], v["victim_slowdown_pct"])
        for k, v in g10["summary"].items() if k.startswith("write@")
    )
    if not pts:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= frac <= x1:
            return round(y0 + (frac - x0) / (x1 - x0) * (y1 - y0), 2)
    return round(pts[0][1] if frac < pts[0][0] else pts[-1][1], 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-iters", type=int, default=200)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g11_consumer_victim")
    import torch
    import torch.nn.functional as F

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = args.batch, args.ctx
    dt = torch.float16
    # victim decode lives on cuda:0 (the CONSUMER / destination of the handoff)
    torch.cuda.set_device(0)
    W = {k: torch.randn(*s, dtype=dt, device="cuda:0") * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
    Kc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:0") * 0.02
    Vc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:0") * 0.02
    x = torch.randn(B, 1, D, dtype=dt, device="cuda:0") * 0.02

    def decode_step():
        q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
        k = (x @ W["k"]).view(B, 1, HKV, HD).transpose(1, 2)
        v = (x @ W["v"]).view(B, 1, HKV, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    total = args.handoff_mb * 1024 * 1024 // 2
    nbytes = total * 2
    src_p = torch.randn(total, dtype=dt, device="cuda:1")                    # holder-side source
    src_host = torch.randn(total, dtype=dt, device="cpu", pin_memory=True)   # host source
    dst_c = torch.empty(total, dtype=dt, device="cuda:0")                    # lands in consumer HBM

    dec_stream = torch.cuda.Stream(device=0)
    s_dev1 = torch.cuda.Stream(device=1)   # holder-issued (push)
    s_dev0 = torch.cuda.Stream(device=0)   # consumer-issued (pull / host_in)

    def copy_for(dest):
        if dest == "peer_in_push":
            return s_dev1, (lambda: dst_c.copy_(src_p, non_blocking=True))
        if dest == "peer_in_pull":
            return s_dev0, (lambda: dst_c.copy_(src_p, non_blocking=True))
        if dest == "host_in":
            return s_dev0, (lambda: dst_c.copy_(src_host, non_blocking=True))
        raise ValueError(dest)

    for _ in range(30):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    def timed_iter():
        with torch.cuda.stream(dec_stream):
            e0.record(dec_stream); decode_step(); e1.record(dec_stream)
        e1.synchronize()
        return e0.elapsed_time(e1)

    def isolated_copy_bw(dest):
        st, op = copy_for(dest)
        ce0 = torch.cuda.Event(enable_timing=True); ce1 = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            with torch.cuda.stream(st):
                op()
        st.synchronize()
        ts = []
        for _ in range(20):
            with torch.cuda.stream(st):
                ce0.record(st); op(); ce1.record(st)
            ce1.synchronize()
            ts.append(ce0.elapsed_time(ce1))
        return nbytes / (statistics.median(ts) / 1e3) / 1e9

    hbm_peak = 2039.0  # A100-SXM4-80GB
    dests = ["peer_in_push", "peer_in_pull", "host_in"]
    results = {}
    for dest in dests:
        bw = isolated_copy_bw(dest)
        slow_seeds = []
        ho_ms_seeds = []
        for _ in range(args.seeds):
            base = statistics.median([timed_iter() for _ in range(args.base_iters)])
            st, op = copy_for(dest)
            cev1 = torch.cuda.Event(enable_timing=True)
            during = []
            t0 = time.time()
            with torch.cuda.stream(st):
                op()
                cev1.record(st)
            guard = 0
            while not st.query() and guard < 200000:
                during.append(timed_iter())
                guard += 1
            cev1.synchronize()
            ho_ms_seeds.append((time.time() - t0) * 1e3)
            if during:
                slow_seeds.append(statistics.median(during) / base - 1)
        frac = bw / hbm_peak
        results[dest] = {
            "copy_bw_gbs": round(bw, 1),
            "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2) if slow_seeds else None,
            "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
            "handoff_ms": round(statistics.mean(ho_ms_seeds), 2),
            "footprint_frac_of_peak": round(frac, 4),
            "g10_write_curve_pred_pct": g10_write_prediction(frac),
        }
        r = results[dest]
        print(f"  {dest:13s} copy_bw={r['copy_bw_gbs']:6.1f}GB/s  victim=+{r['victim_slowdown_pct']:5.2f}% "
              f"(±{r['victim_slowdown_std']})  handoff={r['handoff_ms']:.2f}ms  "
              f"g10-write-pred=+{r['g10_write_curve_pred_pct']}%")

    out = {"_experiment": "g11_consumer_victim", "_is_measured": True,
           "_timing_method": "consumer-decode-stream events; single handoff; clocks locked; no whole-device sync",
           "device": __import__("torch").cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S, "batch": B,
           "handoff_mb": args.handoff_mb, "seeds": args.seeds,
           "victim_side": "consumer (destination GPU0); the copy WRITES the victim's HBM",
           "destinations": results,
           "note": "consumer-side mirror of g3: direction is still a non-effect; the write-landing "
                   "footprint sets the cost, checked against the g10 write-only paced curve.",
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

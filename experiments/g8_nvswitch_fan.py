"""G8 -- NVSwitch fan-in/fan-out: does the direction null survive multi-GPU contention?

The paper establishes (one-to-one, direct NVLink3 bridge and one NVSwitch hop):
push == pull, and the victim cost tracks the copy's holder-side HBM-read footprint.
The open question for switched fabrics (HGX 8-GPU, NVL72) is *fan* arbitration:

  fan-out (1 holder -> K consumers): K consumers simultaneously PULL distinct
      buffers from one busy holder (or the holder PUSHes to all K). The holder's
      aggregate HBM-read footprint grows with K. Does push/pull still agree at
      every K? Does the victim cost track the *aggregate* footprint (cost law),
      or does switch-side queueing privilege one issuer?

  fan-in (K holders -> 1 consumer): K busy holders each hand off to one idle
      consumer. The consumer's HBM write side and switch egress are shared, so
      per-copy bandwidth may drop with K; per the cost law each holder's victim
      cost should drop with its own (reduced) read footprint. We time holder 0.

Requires >= 3 GPUs (ideally an 8-GPU HGX with NVSwitch). Self-contained:
no repo imports, random weights, JSON out.

Usage:  python g8_nvswitch_fan.py [--fans 1,2,4,7] [--seeds 5] [--handoff-mb 512]
"""
from __future__ import annotations
import argparse, json, statistics, subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g8_nvswitch_fan.json"


def topology_forensics():
    """Record the evidence needed to classify the fabric (direct vs NVSwitch)."""
    info = {}
    for key, cmd in {
        "topo_matrix": ["nvidia-smi", "topo", "-m"],
        "nvlink_status_gpu0": ["nvidia-smi", "nvlink", "-s", "-i", "0"],
        "lspci_nvswitch": ["bash", "-c", "lspci -d 10de: 2>/dev/null | grep -i -E 'bridge|switch' || true"],
    }.items():
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # forensics must never kill the experiment
            info[key] = f"<failed: {e}>"
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-iters", type=int, default=200)
    ap.add_argument("--overlap-iters", type=int, default=100)
    ap.add_argument("--fans", type=str, default="")  # e.g. "1,2,4,7"; default = 1,2,4,N-1
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F

    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 3, f"g8 needs >=3 GPUs for a fan (got {n_gpu}); use g1-g4 for 1-to-1"
    fans = [int(x) for x in args.fans.split(",") if x] or [k for k in (1, 2, 4, n_gpu - 1) if k <= n_gpu - 1]
    fans = sorted(set(fans))

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    S, dt = args.ctx, torch.float16

    def make_holder(dev):
        W = {k: torch.randn(*s, dtype=dt, device=dev) * 0.02 for k, s in {
            "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
            "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
        Kc = torch.randn(1, HKV, S, HD, dtype=dt, device=dev) * 0.02
        Vc = torch.randn(1, HKV, S, HD, dtype=dt, device=dev) * 0.02
        x = torch.randn(1, 1, D, dtype=dt, device=dev) * 0.02

        def step():
            q = (x @ W["q"]).view(1, 1, H, HD).transpose(1, 2)
            k = (x @ W["k"]); v = (x @ W["v"])  # noqa: F841  traffic fidelity with g1-g4
            o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
            return (o.transpose(1, 2).reshape(1, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]
        return step

    total = args.handoff_mb * 1024 * 1024 // 2
    nbytes = total * 2

    def measure_victim(holder_dev, dec_stream, decode_step, copy_streams, ops):
        """g3-style ratio: median decode-iter time while the fan is in flight / alone.
        Drained copy streams are re-armed *before* each timed iteration."""
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)

        def timed_iter():
            with torch.cuda.device(holder_dev), torch.cuda.stream(dec_stream):
                e0.record(dec_stream); decode_step(); e1.record(dec_stream)
            e1.synchronize()
            return e0.elapsed_time(e1)

        for _ in range(30):
            with torch.cuda.device(holder_dev), torch.cuda.stream(dec_stream):
                decode_step()
        dec_stream.synchronize()
        base = statistics.median([timed_iter() for _ in range(args.base_iters)])

        during = []
        while len(during) < args.overlap_iters:
            for st, op in zip(copy_streams, ops):
                if st.query():  # drained -> keep the fan saturated
                    with torch.cuda.device(st.device), torch.cuda.stream(st):
                        op()
            during.append(timed_iter())
        for st in copy_streams:
            st.synchronize()
        return statistics.median(during) / base - 1

    def fan_bw(streams, ops):
        """Isolated (no decode) per-stream and aggregate copy bandwidth."""
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in streams]
        per = []
        for _ in range(5):
            for (a, b), st, op in zip(evs, streams, ops):
                with torch.cuda.device(st.device), torch.cuda.stream(st):
                    a.record(st); op(); b.record(st)
            for st in streams:
                st.synchronize()
            per.append([nbytes / (a.elapsed_time(b) / 1e3) / 1e9 for a, b in evs])
        med = [statistics.median(x) for x in zip(*per)]
        return round(sum(med), 1), [round(m, 1) for m in med]

    results = {"fan_out": {}, "fan_in": {}}

    # ---------- fan-out: one busy holder (GPU0) -> K consumers (GPU1..K) ----------
    holder_dev = 0
    torch.cuda.set_device(holder_dev)
    decode_step = make_holder(holder_dev)
    dec_stream = torch.cuda.Stream(device=holder_dev)
    srcs = {k: torch.randn(total, dtype=dt, device=holder_dev) for k in range(max(fans))}
    dsts = {k: torch.empty(total, dtype=dt, device=k + 1) for k in range(max(fans))}

    for K in fans:
        for direction in ("push", "pull"):
            streams = [torch.cuda.Stream(device=(k + 1) if direction == "pull" else holder_dev) for k in range(K)]
            ops = [(lambda k=k: dsts[k].copy_(srcs[k], non_blocking=True)) for k in range(K)]
            agg_bw, per_bw = fan_bw(streams, ops)
            slows = [measure_victim(holder_dev, dec_stream, decode_step, streams, ops)
                     for _ in range(args.seeds)]
            key = f"K{K}_{direction}"
            results["fan_out"][key] = {
                "consumers": K, "direction": direction,
                "aggregate_copy_bw_gbs": agg_bw, "per_consumer_bw_gbs": per_bw,
                "victim_slowdown_pct": round(statistics.mean(slows) * 100, 2),
                "victim_slowdown_std": round(statistics.pstdev(slows) * 100, 2) if len(slows) > 1 else 0.0,
                "victim_slowdown_pct_seeds": [round(s * 100, 2) for s in slows],
            }
            r = results["fan_out"][key]
            print(f"  fan_out {key:9s} agg_bw={agg_bw:7.1f}GB/s  victim=+{r['victim_slowdown_pct']:6.2f}% (±{r['victim_slowdown_std']})")

    # ---------- fan-in: K busy holders (GPU0..K-1) -> 1 idle consumer (last GPU) ----------
    consumer = n_gpu - 1
    fan_in_fans = [k for k in fans if k <= n_gpu - 2]
    if not fan_in_fans:
        print("  fan_in: skipped (needs at least one K <= n_gpu-2)")
    holders = list(range(max(fan_in_fans))) if fan_in_fans else []
    steps = {h: (decode_step if h == 0 else make_holder(h)) for h in holders}
    bg_streams = {h: torch.cuda.Stream(device=h) for h in holders if h != 0}
    h_srcs = {h: (srcs[0] if h == 0 else torch.randn(total, dtype=dt, device=h)) for h in holders}
    h_dsts = {h: torch.empty(total, dtype=dt, device=consumer) for h in holders}
    dec0 = torch.cuda.Stream(device=0)

    for K in fan_in_fans:
        copy_streams = [torch.cuda.Stream(device=h) for h in range(K)]  # push: holder-issued
        ops = [(lambda h=h: h_dsts[h].copy_(h_srcs[h], non_blocking=True)) for h in range(K)]
        agg_bw, per_bw = fan_bw(copy_streams, ops)
        slows = []
        for _ in range(args.seeds):
            for h in range(1, K):  # saturate background holders ~1s ahead
                with torch.cuda.device(h), torch.cuda.stream(bg_streams[h]):
                    for _ in range(2500):
                        steps[h]()
            slows.append(measure_victim(0, dec0, steps[0], copy_streams, ops))
            for h in range(1, K):
                bg_streams[h].synchronize()
        results["fan_in"][f"K{K}_push"] = {
            "holders": K, "direction": "push", "timed_holder": 0,
            "aggregate_copy_bw_gbs": agg_bw, "per_holder_bw_gbs": per_bw,
            "holder0_victim_slowdown_pct": round(statistics.mean(slows) * 100, 2),
            "holder0_victim_slowdown_std": round(statistics.pstdev(slows) * 100, 2) if len(slows) > 1 else 0.0,
            "holder0_victim_slowdown_pct_seeds": [round(s * 100, 2) for s in slows],
        }
        r = results["fan_in"][f"K{K}_push"]
        print(f"  fan_in  K{K}_push   agg_bw={agg_bw:7.1f}GB/s  holder0 victim=+{r['holder0_victim_slowdown_pct']:6.2f}% (±{r['holder0_victim_slowdown_std']})")

    out = {"_experiment": "g8_nvswitch_fan", "_is_measured": True,
           "_timing_method": "G3 ratio protocol; copies kept in flight across the fan; decode-stream events only",
           "device": torch.cuda.get_device_name(0), "n_gpu": n_gpu,
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S,
           "handoff_mb": args.handoff_mb, "seeds": args.seeds, "fans": fans,
           "topology_forensics": topology_forensics(),
           "results": results,
           "questions": ["Does push==pull hold at every K (fan direction null)?",
                          "Does holder victim cost track aggregate read footprint (cost law) under fan-out?",
                          "Does per-copy BW drop and holder cost drop accordingly under fan-in?"],
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

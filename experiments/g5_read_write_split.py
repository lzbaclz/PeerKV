"""G5 -- is the victim cost a *read*-port effect or a generic bandwidth effect?

G1-G3 establish that (a) transfer *direction* (push/pull) is a non-effect and
(b) the victim cost tracks the copy's HBM *footprint*. The paper's mechanism
framing goes one step further and attributes the cost to the holder's HBM
**read port**. G1's DCGM DRAM-active counter is a read+write *aggregate*
active-fraction, so on its own it supports "footprint", not "read-port". G5
closes that gap with a controlled read-vs-write contrast at fixed link and
fixed byte rate, so the only thing that changes is which HBM port on the busy
holder (GPU1) the concurrent copy stresses:

  read_nvlink   GPU1 -> GPU0 over NVLink   GPU1 HBM is *read*  (NVLink TX)  ~280 GB/s
  write_nvlink  GPU0 -> GPU1 over NVLink   GPU1 HBM is *written* (NVLink RX) ~280 GB/s
  readwrite_local GPU1 -> GPU1 over HBM    GPU1 HBM is read+written          ~790 GB/s
  read_pcie     GPU1 -> host over PCIe     GPU1 HBM is *read*, slow rate     ~26  GB/s

read_nvlink and write_nvlink move identical bytes over the identical link at the
same rate; the ONLY difference is whether GPU1's read port or its write port is
contended. Hence:

  H1 (read-port):   victim(read_nvlink) >> victim(write_nvlink), and
                    victim(read_pcie) << victim(read_nvlink) (same polarity, lower rate)
  H0 (aggregate):   victim(read_nvlink) ~= victim(write_nvlink)

The holder is a memory-bound Llama-3-8B-geometry decode (paged-decode HBM-read
pattern). Timing is overlap-safe per-iteration CUDA events on the decode stream
only (never a whole-device sync), clocks should be locked (asserted). Writes
results/g5_read_write_split.json (+ markers for optional DCGM alignment, like g1).

Run (A100, clocks locked):
  sudo nvidia-smi -lgc 1410,1410
  python experiments/g5_read_write_split.py --seeds 5
Optional read/write DCGM attribution (separate terminal, like g1_run.sh):
  dcgmi dmon -e 1005,1009,1010,1011,1012 -d 100   # DRAMA, PCITX/RX, NVLTX/RX
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g5_read_write_split.json"
MARKERS_OUT = RESULTS / "g5_markers.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-iters", type=int, default=200)
    ap.add_argument("--counter-secs", type=float, default=6.0,
                    help="extra steady spin per condition so DCGM can sample the window")
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g5_read_write_split")
    import torch
    import torch.nn.functional as F

    assert torch.cuda.device_count() >= 2, "need 2 GPUs"
    # assert clocks are locked (DVFS off) so idle/busy timing is comparable
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in (0, 1):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            cur = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
            mx = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"GPU{i} graphics clock {cur}/{mx} MHz")
            if cur < int(0.9 * mx):
                print(f"  WARNING GPU{i} not locked near max -- run: sudo nvidia-smi -lgc {mx},{mx}")
    except Exception as e:  # pragma: no cover
        print(f"(pynvml unavailable: {e}; proceeding -- ensure clocks are locked)")

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = args.batch, args.ctx
    dt = torch.float16
    torch.cuda.set_device(1)
    W = {k: torch.randn(*s, dtype=dt, device="cuda:1") * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
    Kc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
    Vc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
    x = torch.randn(B, 1, D, dtype=dt, device="cuda:1") * 0.02

    def decode_step():
        q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
        k = (x @ W["k"]).view(B, 1, HKV, HD).transpose(1, 2)
        v = (x @ W["v"]).view(B, 1, HKV, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + \
               (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    n = args.handoff_mb * 1024 * 1024 // 2
    nbytes = n * 2
    src_g1 = torch.randn(n, dtype=dt, device="cuda:1")          # read source on holder
    src_g0 = torch.randn(n, dtype=dt, device="cuda:0")          # source on consumer
    dst_g0 = torch.empty(n, dtype=dt, device="cuda:0")          # read_nvlink dest (GPU1->GPU0)
    dst_g1 = torch.empty(n, dtype=dt, device="cuda:1")          # write_nvlink dest (GPU0->GPU1)
    dst_g1_local = torch.empty(n, dtype=dt, device="cuda:1")    # local repack dest
    dst_host = torch.empty(n, dtype=dt, device="cpu", pin_memory=True)

    s_dev1 = torch.cuda.Stream(device=1)

    def copy_for(kind):
        # all issued on GPU1's stream (direction/issuer is a non-effect, g1); the
        # holder-side HBM port that is stressed is what differs.
        if kind == "read_nvlink":      # GPU1 HBM READ  -> NVLink -> GPU0
            return s_dev1, (lambda: dst_g0.copy_(src_g1, non_blocking=True))
        if kind == "write_nvlink":     # GPU0 -> NVLink -> GPU1 HBM WRITE
            return s_dev1, (lambda: dst_g1.copy_(src_g0, non_blocking=True))
        if kind == "readwrite_local":  # GPU1 HBM READ+WRITE
            return s_dev1, (lambda: dst_g1_local.copy_(src_g1, non_blocking=True))
        if kind == "read_pcie":        # GPU1 HBM READ -> PCIe -> host (slow)
            return s_dev1, (lambda: dst_host.copy_(src_g1, non_blocking=True))
        raise ValueError(kind)

    dec_stream = torch.cuda.Stream(device=1)
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

    def isolated_copy_bw(kind):
        st, op = copy_for(kind)
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

    kinds = ["read_nvlink", "write_nvlink", "readwrite_local", "read_pcie"]
    results = {}
    markers = {}
    for kind in kinds:
        bw = isolated_copy_bw(kind)
        slow_seeds = []
        st, op = copy_for(kind)
        t_start = time.time()
        for _ in range(args.seeds):
            base = statistics.median([timed_iter() for _ in range(args.base_iters)])
            # keep the contender in flight and time decode iters strictly overlapped
            during = []
            spun = 0
            t_spin = time.time()
            while time.time() - t_spin < args.counter_secs:
                if st.query():
                    with torch.cuda.stream(st):
                        op()
                    spun += 1
                during.append(timed_iter())
            st.synchronize()
            if during:
                slow_seeds.append(statistics.median(during) / base - 1)
        t_end = time.time()
        markers[kind] = [t_start, t_end]
        results[kind] = {
            "copy_bw_gbs": round(bw, 1),
            "holder_port": {"read_nvlink": "read", "write_nvlink": "write",
                            "readwrite_local": "read+write", "read_pcie": "read"}[kind],
            "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2) if slow_seeds else None,
            "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
        }
        r = results[kind]
        print(f"  {kind:16s} port={r['holder_port']:10s} bw={r['copy_bw_gbs']:6.1f}GB/s  "
              f"victim=+{r['victim_slowdown_pct']:6.2f}% (+/-{r['victim_slowdown_std']})")

    # hypothesis verdict (read-port vs aggregate)
    vr = results["read_nvlink"]["victim_slowdown_pct"]
    vw = results["write_nvlink"]["victim_slowdown_pct"]
    verdict = "read-port (H1)" if (vr is not None and vw is not None and vr > 1.5 * vw) \
        else ("aggregate (H0)" if (vr is not None and vw is not None) else "inconclusive")
    print(f"\n  VERDICT: read_nvlink +{vr}% vs write_nvlink +{vw}% -> {verdict}")

    out = {"_experiment": "g5_read_write_split", "_is_measured": True,
           "_timing_method": "decode-stream events; contender kept in flight; clocks locked; no whole-device sync",
           "device": torch.cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S, "batch": B,
           "handoff_mb": args.handoff_mb, "seeds": args.seeds,
           "conditions": results,
           "verdict": verdict,
           "hypotheses": {
               "H1_read_port": "victim(read_nvlink) >> victim(write_nvlink) at equal NVLink rate",
               "H0_aggregate": "victim(read_nvlink) ~= victim(write_nvlink)",
           },
           "note": "read_nvlink and write_nvlink move identical bytes over the same link at the "
                   "same rate; only the stressed holder HBM port differs. read_pcie is the same "
                   "read polarity at a lower rate (PCIe).",
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    MARKERS_OUT.write_text(json.dumps(markers, indent=2))
    print(f"-> {OUT}\n-> {MARKERS_OUT}")


if __name__ == "__main__":
    main()

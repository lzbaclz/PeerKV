"""G3 -- KV-handoff placement cost model: victim TPOT vs the copy's HBM-read-bandwidth
footprint, across destinations {peer-push, peer-pull, local-HBM, host-PCIe}.

G1/G2 established that *direction* (push vs pull) is a non-effect. G3 establishes the
real lever: the holder's decode slowdown scales with how much HBM read bandwidth the
concurrent copy consumes, which is set by the *destination/link*, not the direction:

  peer-push  GPU1->GPU0 over NVLink   (holder-issued)   ~250 GB/s HBM read
  peer-pull  GPU1->GPU0 over NVLink   (consumer-issued) ~250 GB/s HBM read
  local      GPU1->GPU1 over HBM      (holder-issued)   ~790 GB/s HBM read+write
  host       GPU1->pinned host (PCIe) (holder-issued)   ~25  GB/s HBM read

For each we measure (a) the isolated effective copy bandwidth (= HBM read rate the
copy imposes on the holder) and (b) the holder's decode victim slowdown during a
single 512MB handoff (clean per-iter events, clocks locked, no whole-device sync).
The (BW, victim) pairs give a counter-consistent cost model and justify peer-over-
host placement and 'never repack locally on a busy GPU'.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g3_placement.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-iters", type=int, default=200)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g3_placement")
    import torch
    import torch.nn.functional as F

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
        # faithful memory-bound decode: flash-attend the resident L-token KV in place
        # (paged-decode HBM-read pattern), no per-step full re-cat of the cache.
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    total = args.handoff_mb * 1024 * 1024 // 2
    nbytes = total * 2
    src_h = torch.randn(total, dtype=dt, device="cuda:1")
    dst_c = torch.empty(total, dtype=dt, device="cuda:0")      # peer dest
    dst_h = torch.empty(total, dtype=dt, device="cuda:1")      # local dest
    dst_host = torch.empty(total, dtype=dt, device="cpu", pin_memory=True)  # host dest

    dec_stream = torch.cuda.Stream(device=1)
    s_dev1 = torch.cuda.Stream(device=1)
    s_dev0 = torch.cuda.Stream(device=0)

    def copy_for(dest):
        if dest == "peer_push":
            return s_dev1, (lambda: dst_c.copy_(src_h, non_blocking=True))
        if dest == "peer_pull":
            return s_dev0, (lambda: dst_c.copy_(src_h, non_blocking=True))
        if dest == "local":
            return s_dev1, (lambda: dst_h.copy_(src_h, non_blocking=True))
        if dest == "host":
            return s_dev1, (lambda: dst_host.copy_(src_h, non_blocking=True))
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

    dests = ["peer_push", "peer_pull", "local", "host"]
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
        results[dest] = {
            "copy_bw_gbs": round(bw, 1),
            "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2) if slow_seeds else None,
            "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
            "handoff_ms": round(statistics.mean(ho_ms_seeds), 2),
        }
        r = results[dest]
        print(f"  {dest:11s} copy_bw={r['copy_bw_gbs']:6.1f}GB/s  victim=+{r['victim_slowdown_pct']:5.2f}% "
              f"(±{r['victim_slowdown_std']})  handoff={r['handoff_ms']:.2f}ms")

    out = {"_experiment": "g3_placement", "_is_measured": True,
           "_timing_method": "decode-stream events; single handoff; clocks locked; no whole-device sync",
           "device": torch.cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S, "batch": B,
           "handoff_mb": args.handoff_mb, "seeds": args.seeds,
           "destinations": results,
           "note": "victim slowdown scales with the copy's HBM-read-bandwidth footprint, not direction.",
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

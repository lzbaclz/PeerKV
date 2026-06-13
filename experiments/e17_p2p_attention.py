"""e17 -- end-to-end decode-step latency: NVLink-tiered vs host-offload (RQ2 MEASURED).

RUN ON THE DUAL-A100 BOX. This is the measured counterpart of e16's cost-model
prediction. We lay out a long-context KV cache as blocks: the hottest
`--local-blocks` stay on the compute GPU (cuda:0, T0); the overflow goes either
to the *peer GPU* over NVLink (T1, ours) or to *host DRAM* over PCIe (T2,
FlexGen/OrchKvCache line). Each decode step fetches the non-local blocks back to
cuda:0 and runs scaled-dot-product attention; we time the whole step. The
difference between the two placements is the fetch link (NVLink vs PCIe), i.e.
exactly the ~10x bandwidth gap the paper exploits.

Authored on a Mac with no CUDA; the no-GPU / <2-GPU path writes a placeholder.
On the dual-A100 box:

    pip install "torch>=2.4"
    python experiments/e17_p2p_attention.py --n-blocks 512 --local-blocks 64

Writes experiments/results/p2p_attention.json. Honest note: this is a
no-prefetch upper bound (every overflow block is fetched each step); a
one-step-ahead prefetch (described in implementation.tex) would hide latency
behind compute -- a separate measurement. Send the JSON back.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "p2p_attention.json"


def _write(res: dict) -> None:
    res["_experiment"] = "e17_p2p_attention"
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=512, help="total KV blocks (context)")
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--local-blocks", type=int, default=64, help="blocks kept on cuda:0")
    ap.add_argument("--heads", type=int, default=8)       # n_kv_heads
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    res = {k: getattr(args, k) for k in
           ("n_blocks", "block_tokens", "local_blocks", "heads", "head_dim", "trials")}
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # noqa: BLE001
        res.update({"_is_measured": False, "note": f"torch not importable: {e}"})
        _write(res); return

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        res.update({"_is_measured": False,
                    "note": f"need >=2 CUDA GPUs (have {ngpu}); run on the dual-A100 box"})
        _write(res); print(f"  only {ngpu} GPU(s) -- wrote placeholder."); return

    H, T, D = args.heads, args.block_tokens, args.head_dim
    N, L = args.n_blocks, min(args.local_blocks, args.n_blocks)
    n_spill = N - L
    scale = 1.0 / (D ** 0.5)
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))

    def mk(dev, pin=False):
        t = torch.randn(1, H, T, D, dtype=torch.float16,
                        device=("cpu" if dev == "host" else dev))
        return t.pin_memory() if (dev == "host" and pin) else t

    q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")
    # local (hot) blocks live on the compute GPU for every placement
    Kloc = [mk("cuda:0") for _ in range(L)]
    Vloc = [mk("cuda:0") for _ in range(L)]

    def bench(spill_dev):
        # overflow blocks resident on the spill device (peer GPU or host)
        Ksp = [mk(spill_dev, pin=True) for _ in range(n_spill)]
        Vsp = [mk(spill_dev, pin=True) for _ in range(n_spill)]

        def step():
            Ks = [k for k in Kloc] + [k.to("cuda:0", non_blocking=True) for k in Ksp]
            Vs = [v for v in Vloc] + [v.to("cuda:0", non_blocking=True) for v in Vsp]
            K = torch.cat(Ks, dim=2); V = torch.cat(Vs, dim=2)
            return F.scaled_dot_product_attention(q, K, V, scale=scale)

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            o = step(); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        del Ksp, Vsp
        torch.cuda.empty_cache()
        return statistics.median(ts) * 1e3  # ms/step

    # all-local reference (no spill): only meaningful if it fits HBM
    def bench_all_local():
        Kall = Kloc + [mk("cuda:0") for _ in range(n_spill)]
        Vall = Vloc + [mk("cuda:0") for _ in range(n_spill)]

        def step():
            return F.scaled_dot_product_attention(
                q, torch.cat(Kall, dim=2), torch.cat(Vall, dim=2), scale=scale)
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            step(); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        del Kall, Vall
        torch.cuda.empty_cache()
        return statistics.median(ts) * 1e3

    t_nv = bench("cuda:1")    # NVLink-tiered (ours)
    t_host = bench("host")    # host-offload baseline
    t_loc = bench_all_local()  # reference

    res.update({
        "_is_measured": True,
        "kind": "end_to_end_decode_step",
        "device": torch.cuda.get_device_name(0),
        "n_gpus": ngpu, "peer_access_enabled": p2p,
        "spill_blocks": n_spill,
        "all_local_ms": t_loc,
        "nvlink_tiered_ms": t_nv,
        "host_offload_ms": t_host,
        "nvlink_speedup_vs_host": (t_host / t_nv) if t_nv else None,
        "note": ("no-prefetch upper bound (every overflow block fetched each "
                 "step); prefetch would hide latency -- separate measurement. "
                 "If peer_access_enabled is False, the NVLink path fell back to "
                 "a host bounce; check NVLink topology."),
    })
    _write(res)
    print(f"  all_local={t_loc:.2f}ms  nvlink={t_nv:.2f}ms  host={t_host:.2f}ms  "
          f"speedup={t_host/t_nv:.2f}x  (p2p={p2p})")


if __name__ == "__main__":
    main()

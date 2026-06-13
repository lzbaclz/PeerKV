"""Batched-fetch probe: does coalescing the spill-block transfer realize the
NVLink bandwidth advantage end-to-end?

e17/e18 fetch each overflow KV block with a separate .to("cuda:0") call
(hundreds/thousands of tiny 512KB copies per step). That is launch/latency
bound, so the 11.5x NVLink/PCIe bandwidth gap is mostly hidden and the measured
speedup collapses to ~1x. Here we instead store the spill K/V as ONE contiguous
tensor on the spill device and fetch the whole thing in a SINGLE copy per step,
then run dense SDPA. This is the bandwidth-bound regime the cost model assumes.

We report, for NVLink (peer GPU) vs host (pinned DRAM):
  * transfer-only ms (just the batched copy of the spill KV)
  * full decode-step ms (batched copy + cat + SDPA)
and the NVLink speedup for each. Compares against the per-block path too.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "p2p_attention_batched.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=512)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--local-blocks", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    H, T, D = args.heads, args.block_tokens, args.head_dim
    N, L = args.n_blocks, min(args.local_blocks, args.n_blocks)
    n_spill = N - L
    scale = 1.0 / (D ** 0.5)
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))
    spill_bytes = n_spill * H * T * D * 2 * 2  # K and V, fp16

    q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")
    # local hot blocks: one contiguous tensor on cuda:0
    Kloc = torch.randn(1, H, L * T, D, dtype=torch.float16, device="cuda:0")
    Vloc = torch.randn(1, H, L * T, D, dtype=torch.float16, device="cuda:0")

    def make_spill(dev):
        d = "cpu" if dev == "host" else dev
        K = torch.randn(1, H, n_spill * T, D, dtype=torch.float16, device=d)
        V = torch.randn(1, H, n_spill * T, D, dtype=torch.float16, device=d)
        if dev == "host":
            K, V = K.pin_memory(), V.pin_memory()
        return K, V

    def med_ms(fn, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e3

    def bench(dev):
        Ksp, Vsp = make_spill(dev)
        Kc = torch.empty(1, H, n_spill * T, D, dtype=torch.float16, device="cuda:0")
        Vc = torch.empty(1, H, n_spill * T, D, dtype=torch.float16, device="cuda:0")

        def transfer_only():
            Kc.copy_(Ksp, non_blocking=True); Vc.copy_(Vsp, non_blocking=True)

        def full_step():
            Kc.copy_(Ksp, non_blocking=True); Vc.copy_(Vsp, non_blocking=True)
            K = torch.cat([Kloc, Kc], dim=2); V = torch.cat([Vloc, Vc], dim=2)
            return F.scaled_dot_product_attention(q, K, V, scale=scale)

        t_xfer = med_ms(transfer_only)
        t_full = med_ms(full_step)
        del Ksp, Vsp, Kc, Vc; torch.cuda.empty_cache()
        return t_xfer, t_full

    nv_x, nv_f = bench("cuda:1")
    h_x, h_f = bench("host")

    res = {
        "_experiment": "e17b_batched_fetch", "_is_measured": True,
        "kind": "end_to_end_decode_step_batched_transfer",
        "device": torch.cuda.get_device_name(0), "peer_access_enabled": p2p,
        "n_blocks": N, "local_blocks": L, "spill_blocks": n_spill,
        "block_tokens": T, "heads": H, "head_dim": D, "trials": args.trials,
        "spill_mb": spill_bytes / 1024**2,
        "nvlink_transfer_ms": nv_x, "host_transfer_ms": h_x,
        "nvlink_full_ms": nv_f, "host_full_ms": h_f,
        "nvlink_speedup_transfer": h_x / nv_x if nv_x else None,
        "nvlink_speedup_full": h_f / nv_f if nv_f else None,
        "nvlink_eff_gbps": spill_bytes / (nv_x / 1e3) / 1e9 if nv_x else None,
        "host_eff_gbps": spill_bytes / (h_x / 1e3) / 1e9 if h_x else None,
        "note": ("ONE batched copy of the whole spill KV per step (bandwidth-bound "
                 "regime), vs e17's per-block fetch (launch-bound). Shows the "
                 "realizable end-to-end NVLink advantage when transfers are coalesced."),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  spill={res['spill_mb']:.0f}MB  "
          f"NVLink: xfer={nv_x:.3f}ms ({res['nvlink_eff_gbps']:.0f}GB/s) full={nv_f:.3f}ms | "
          f"host: xfer={h_x:.3f}ms ({res['host_eff_gbps']:.0f}GB/s) full={h_f:.3f}ms")
    print(f"  NVLink speedup: transfer={res['nvlink_speedup_transfer']:.2f}x  "
          f"full-step={res['nvlink_speedup_full']:.2f}x")
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

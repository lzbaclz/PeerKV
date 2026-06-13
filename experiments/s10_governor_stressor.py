"""s10 ingress stressor: sustained KV-style ingress INTO the vLLM GPU.

The e2e anchor's traffic generator.  Unlike e2e_handoff_stressor.py (which
reads the vLLM GPU = holder side), this WRITES the vLLM GPU's HBM
(GPU0 -> GPU1, vLLM serves on physical GPU1) -- the receiver-ingress
headline regime of s2, now with a real engine as the victim.

  --arm unpaced   saturating back-to-back 512MB copies (NIXL-default analog)
  --arm governed  Governor(ff, worst-bucket, native pacer) admitting the
                  same stream against eps -- the contract under test is that
                  the engine's client-measured TPOT inflation stays ~<=eps
                  even though the calibration victim was a synthetic decode,
                  not this engine.

Runs as a separate process with its own CUDA context (vLLM is started with
CUDA_VISIBLE_DEVICES=1 and sees one GPU; we see both).  SIGTERM-clean.
"""
from __future__ import annotations

import argparse
import signal
import threading
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["unpaced", "governed"], required=True)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--secs", type=float, default=70.0)
    ap.add_argument("--eps-pct", type=float, default=5.0)
    ap.add_argument("--chunk-mb", type=int, default=64)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("s10_governor_stressor")   # honors PEERKV_SKIP_IDLE_PROBE

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    import torch
    from umallm.governor import Governor, GovernorCalibration, TransferRequest
    from pathlib import Path

    n = args.mb * (1 << 20) // 2
    src = torch.randn(n, dtype=torch.float16, device="cuda:0")
    dst = torch.empty(n, dtype=torch.float16, device="cuda:1")
    t0 = time.time()
    delivered = 0

    if args.arm == "unpaced":
        from umallm.governor.native import NativePacedCopier, available
        from umallm.governor.pacer import PacedCopier
        copier = (NativePacedCopier(device=0) if available()
                  else PacedCopier(device=0))
        while not stop.is_set() and time.time() - t0 < args.secs:
            r = copier.run(src, dst, lambda: 0.0, args.chunk_mb << 20,
                           cancel=stop, unpaced=True)
            delivered += r.bytes_launched
    else:
        cal = GovernorCalibration.load(
            Path(__file__).resolve().parent / "results" / "governor_calib.json")
        gov = Governor(cal, eps_holder_pct=args.eps_pct,
                       eps_receiver_pct=args.eps_pct, mode="ff",
                       chunk_mb=args.chunk_mb, pacer="auto")
        while not stop.is_set() and time.time() - t0 < args.secs:
            h = gov.submit(TransferRequest(src=src, dst=dst, src_dev=0,
                                           dst_dev=1, route="peer"))
            while not h.done.wait(0.05):
                if stop.is_set():
                    break
            if h.result is not None:
                delivered += h.result.bytes_launched
        gov.shutdown()

    dt = time.time() - t0
    print(f"{args.arm} ingress stressor: {delivered/1e9:.1f} GB in {dt:.1f}s "
          f"(~{delivered/dt/1e9:.1f} GB/s sustained into the vLLM GPU)")


if __name__ == "__main__":
    main()

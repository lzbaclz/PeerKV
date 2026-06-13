"""e2e handoff stressor: a standalone process that runs a sustained 512MB cross-GPU
KV-style handoff to contend with a real vLLM decoder running on physical GPU1.

Direction semantics (holder = GPU1, the GPU vLLM serves on; consumer = GPU0):
  push  : GPU1 issues  GPU1->GPU0  (holder-issued write over NVLink)
  pull  : GPU0 issues  GPU1->GPU0  (consumer-issued remote read over NVLink)
  host  : GPU1 issues  GPU1->pinned host (PCIe offload)
  local : GPU1 issues  GPU1->GPU1  (full-rate local HBM repack; the "brutal" arm of
          the cost law -- proves the +127% magnitude reproduces inside a real engine)
  idle  : no copy (baseline control; just sleeps)

Run as a separate process so it has its own CUDA context seeing both physical GPUs,
while vLLM runs with CUDA_VISIBLE_DEVICES=1. Clocks should be locked by the runner.
Stops on SIGTERM or after --secs.
"""
from __future__ import annotations
import argparse, signal, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", choices=["push", "pull", "host", "local", "idle"], required=True)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--secs", type=float, default=60.0)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("e2e_handoff_stressor")

    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(v=True))

    if args.dir == "idle":
        t0 = time.time()
        while not stop["v"] and time.time() - t0 < args.secs:
            time.sleep(0.1)
        print("idle stressor done")
        return

    import torch
    n = args.mb * 1024 * 1024 // 2
    holder = "cuda:1"      # physical GPU1 = the vLLM holder
    consumer = "cuda:0"
    src_h = torch.randn(n, dtype=torch.float16, device=holder)
    if args.dir in ("push", "pull"):
        dst = torch.empty(n, dtype=torch.float16, device=consumer)
        issue_dev = 1 if args.dir == "push" else 0
        stream = torch.cuda.Stream(device=issue_dev)
        def op():
            with torch.cuda.stream(stream):
                dst.copy_(src_h, non_blocking=True)
    elif args.dir == "local":  # full-rate GPU1->GPU1 HBM repack (the brutal arm)
        dst = torch.empty(n, dtype=torch.float16, device=holder)
        stream = torch.cuda.Stream(device=1)
        def op():
            with torch.cuda.stream(stream):
                dst.copy_(src_h, non_blocking=True)
    else:  # host
        dst = torch.empty(n, dtype=torch.float16, device="cpu", pin_memory=True)
        stream = torch.cuda.Stream(device=1)
        def op():
            with torch.cuda.stream(stream):
                dst.copy_(src_h, non_blocking=True)

    # keep the copy stream saturated: refill whenever it drains (overlap-safe pattern)
    for _ in range(3):
        op()
    t0 = time.time()
    nbytes = n * 2
    count = 0
    while not stop["v"] and time.time() - t0 < args.secs:
        if stream.query():
            op(); count += 1
        else:
            time.sleep(0.0005)
    stream.synchronize()
    dt = time.time() - t0
    print(f"{args.dir} stressor: {count} copies in {dt:.1f}s "
          f"(~{count * nbytes / dt / 1e9:.0f} GB/s sustained)")


if __name__ == "__main__":
    main()

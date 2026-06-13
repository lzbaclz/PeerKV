"""G12b -- occupancy control for g12: decompose the SM-push cost.

g12 measured sm_push (holder-launched copy kernel) at +48.8% victim vs sm_pull
(consumer-launched) at +8.5%. Reviewers ask: how much of the +48.8% is SM
*occupancy* theft vs the kernel's memory traffic? Two controls on the holder,
same 32-block x 256-thread footprint, window-matched to the ~2.7ms copy:

  spin       pure-compute FMA loop, no global memory traffic  -> occupancy only
  local_d2d  same grid-stride float4 copy, src+dst both local -> occupancy + local R/W

With g12's sm_push (occupancy + local read + remote write) this gives the
decomposition. Same g3 victim protocol (per-iteration events, locked clocks).
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g12b_occupancy.json"

CUDA_SRC = r"""
__global__ void copy_kernel(const float4* __restrict__ src,
                            float4* __restrict__ dst, long long n4) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long stride = gridDim.x * (long long)blockDim.x;
  for (; i < n4; i += stride) dst[i] = src[i];
}

__global__ void spin_kernel(float* out, long long iters) {
  float a = threadIdx.x * 1.000001f, b = blockIdx.x * 0.999999f;
  for (long long i = 0; i < iters; ++i) { a = fmaf(a, b, 1.0f); b = fmaf(b, a, -1.0f); }
  if (threadIdx.x == 1024) out[0] = a + b;  // never true: keep the loop live
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

void enable_peer(int a, int b) {
  cudaSetDevice(a); cudaDeviceEnablePeerAccess(b, 0); cudaGetLastError();
  cudaSetDevice(b); cudaDeviceEnablePeerAccess(a, 0); cudaGetLastError();
}

void sm_copy(torch::Tensor dst, torch::Tensor src, int launch_dev, int blocks) {
  long long n4 = src.numel() * src.element_size() / 16;
  cudaSetDevice(launch_dev);
  auto stream = c10::cuda::getCurrentCUDAStream(launch_dev);
  copy_kernel<<<blocks, 256, 0, stream.stream()>>>(
      reinterpret_cast<const float4*>(src.data_ptr()),
      reinterpret_cast<float4*>(dst.data_ptr()), n4);
}

void sm_spin(torch::Tensor out, int launch_dev, int blocks, long long iters) {
  cudaSetDevice(launch_dev);
  auto stream = c10::cuda::getCurrentCUDAStream(launch_dev);
  spin_kernel<<<blocks, 256, 0, stream.stream()>>>(out.data_ptr<float>(), iters);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("enable_peer", &enable_peer);
  m.def("sm_copy", &sm_copy);
  m.def("sm_spin", &sm_spin);
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-iters", type=int, default=200)
    ap.add_argument("--blocks", type=int, default=32)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g12b_occupancy")
    import torch
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load_inline

    ext = load_inline(name="g12b_occ_ext", cpp_sources="", cuda_sources=CUDA_SRC,
                      with_cuda=True, verbose=False)
    ext.enable_peer(0, 1)

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = 1, args.ctx
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
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    total = args.handoff_mb * 1024 * 1024 // 2
    src_l = torch.randn(total, dtype=dt, device="cuda:1")
    dst_l = torch.empty(total, dtype=dt, device="cuda:1")   # local D2D dest
    spin_out = torch.zeros(1, dtype=torch.float32, device="cuda:1")

    work = torch.cuda.Stream(device=1)
    dec_stream = torch.cuda.Stream(device=1)

    # calibrate spin iters to match the ~2.7ms copy window
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    iters = 200000
    with torch.cuda.device(1), torch.cuda.stream(work):
        e0.record(work); ext.sm_spin(spin_out, 1, args.blocks, iters); e1.record(work)
    e1.synchronize()
    ms = e0.elapsed_time(e1)
    iters = int(iters * 2.7 / ms)
    with torch.cuda.device(1), torch.cuda.stream(work):
        e0.record(work); ext.sm_spin(spin_out, 1, args.blocks, iters); e1.record(work)
    e1.synchronize()
    print(f"  spin calibrated: {iters} iters -> {e0.elapsed_time(e1):.2f}ms (target 2.7)")

    def op_for(kind):
        if kind == "spin":
            return lambda: ext.sm_spin(spin_out, 1, args.blocks, iters)
        if kind == "local_d2d":
            return lambda: ext.sm_copy(dst_l, src_l, 1, args.blocks)
        raise ValueError(kind)

    for _ in range(30):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()
    d0 = torch.cuda.Event(enable_timing=True); d1 = torch.cuda.Event(enable_timing=True)

    def timed_iter():
        with torch.cuda.device(1), torch.cuda.stream(dec_stream):
            d0.record(dec_stream); decode_step(); d1.record(dec_stream)
        d1.synchronize()
        return d0.elapsed_time(d1)

    results = {}
    for kind in ("spin", "local_d2d"):
        op = op_for(kind)
        slow_seeds, win_ms = [], []
        for _ in range(args.seeds):
            base = statistics.median([timed_iter() for _ in range(args.base_iters)])
            during = []
            t0 = time.time()
            with torch.cuda.device(1), torch.cuda.stream(work):
                op()
            guard = 0
            while not work.query() and guard < 200000:
                during.append(timed_iter())
                guard += 1
            work.synchronize()
            win_ms.append((time.time() - t0) * 1e3)
            if during:
                slow_seeds.append(statistics.median(during) / base - 1)
        results[kind] = {
            "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2) if slow_seeds else None,
            "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
            "window_ms": round(statistics.mean(win_ms), 2),
            "kernel_blocks": args.blocks,
        }
        r = results[kind]
        print(f"  {kind:10s} victim=+{r['victim_slowdown_pct']:5.2f}% (±{r['victim_slowdown_std']})  window={r['window_ms']:.2f}ms")

    out = {"_experiment": "g12b_occupancy", "_is_measured": True,
           "_timing_method": "decode-stream events; single window; clocks locked; no whole-device sync",
           "device": __import__("torch").cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S,
           "spin_iters": iters, "handoff_mb": args.handoff_mb, "seeds": args.seeds,
           "note": "controls for g12: spin = pure SM occupancy (32 blocks, no memory); "
                   "local_d2d = same copy kernel, src+dst both holder-local. "
                   "sm_push (+48.8%) = occupancy + local read + remote write (g12_sm_copy.json).",
           "results": results,
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

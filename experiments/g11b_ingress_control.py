"""G11b -- consumer-side paced-write control for g11 (both reviewers' top ask).

g11 measured the consumer absorbing a 512MB NVLink-ingress write stream
(~280 GB/s) at +23.5% victim cost -- above the g10 locally-paced write curve
(~+11.8% interpolated). The hypothesis "NVLink-ingress writes cost more per
byte than locally generated writes" was marked future work. G11b tests it
directly: on the SAME victim GPU (cuda:0, g11's consumer), generate LOCAL
write-only traffic with a throttled kernel calibrated to the same ~280 GB/s
sustained, and measure the same decode victim.

  matched-rate local writes  vs  g11 peer_in (~280 GB/s ingress writes)

If local ~= ingress at matched rate, the 2.5x consumer premium is pure write
footprint; if local < ingress, ingress writes are per-byte heavier (memory
controller / link-credit path), confirming the hypothesis.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g11b_ingress_control.json"

CUDA_SRC = r"""
__global__ void write_kernel(float4* __restrict__ dst, long long n4, int passes) {
  const float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
  long long stride = gridDim.x * (long long)blockDim.x;
  for (int p = 0; p < passes; ++p)
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n4; i += stride)
      dst[i] = v;
}

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

void sm_write(torch::Tensor dst, int launch_dev, int blocks, int passes) {
  long long n4 = dst.numel() * dst.element_size() / 16;
  cudaSetDevice(launch_dev);
  auto stream = c10::cuda::getCurrentCUDAStream(launch_dev);
  write_kernel<<<blocks, 256, 0, stream.stream()>>>(
      reinterpret_cast<float4*>(dst.data_ptr()), n4, passes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sm_write", &sm_write); }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--target-gbs", type=float, default=280.0)
    ap.add_argument("--buf-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-iters", type=int, default=200)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g11b_ingress_control")
    import torch
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load_inline

    ext = load_inline(name="g11b_ctl_ext", cpp_sources="", cuda_sources=CUDA_SRC,
                      with_cuda=True, verbose=False)

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = 1, args.ctx
    dt = torch.float16
    # victim decode on cuda:0 == g11's consumer
    torch.cuda.set_device(0)
    W = {k: torch.randn(*s, dtype=dt, device="cuda:0") * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
    Kc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:0") * 0.02
    Vc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:0") * 0.02
    x = torch.randn(B, 1, D, dtype=dt, device="cuda:0") * 0.02

    def decode_step():
        q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
        k = (x @ W["k"]); v = (x @ W["v"])  # noqa
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    buf = torch.empty(args.buf_mb * 1024 * 1024 // 2, dtype=dt, device="cuda:0")
    wstream = torch.cuda.Stream(device=0)
    dec_stream = torch.cuda.Stream(device=0)
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)

    # calibrate block count: find smallest blocks whose SUSTAINED write rate ~ target
    def measure_write_gbs(blocks):
        nbytes = buf.numel() * 2
        with torch.cuda.device(0), torch.cuda.stream(wstream):
            ext.sm_write(buf, 0, blocks, 1)
        wstream.synchronize()
        with torch.cuda.device(0), torch.cuda.stream(wstream):
            e0.record(wstream); ext.sm_write(buf, 0, blocks, 4); e1.record(wstream)
        e1.synchronize()
        return 4 * nbytes / (e0.elapsed_time(e1) / 1e3) / 1e9

    cal = {}
    blocks = None
    for b in (1, 2, 3, 4, 6, 8):
        g = measure_write_gbs(b)
        cal[b] = round(g, 1)
        if blocks is None and g >= args.target_gbs * 0.9:
            blocks = b
    blocks = blocks or 2
    print(f"  write-rate calibration: {cal} -> using {blocks} blocks "
          f"({cal[blocks]} GB/s sustained, target {args.target_gbs})")

    for _ in range(30):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()
    d0 = torch.cuda.Event(enable_timing=True); d1 = torch.cuda.Event(enable_timing=True)

    def timed_iter():
        with torch.cuda.device(0), torch.cuda.stream(dec_stream):
            d0.record(dec_stream); decode_step(); d1.record(dec_stream)
        d1.synchronize()
        return d0.elapsed_time(d1)

    # window-match g11: one 512MB-equivalent write burst, victim timed while in flight
    passes = max(1, round(512 / args.buf_mb))
    slow_seeds, win_ms, eff_gbs = [], [], []
    for _ in range(args.seeds):
        base = statistics.median([timed_iter() for _ in range(args.base_iters)])
        during = []
        t0 = time.time()
        with torch.cuda.device(0), torch.cuda.stream(wstream):
            e0.record(wstream); ext.sm_write(buf, 0, blocks, passes); e1.record(wstream)
        guard = 0
        while not wstream.query() and guard < 200000:
            during.append(timed_iter())
            guard += 1
        e1.synchronize()
        win_ms.append((time.time() - t0) * 1e3)
        ms = e0.elapsed_time(e1)
        eff_gbs.append(passes * buf.numel() * 2 / (ms / 1e3) / 1e9)
        if during:
            slow_seeds.append(statistics.median(during) / base - 1)

    res = {
        "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2),
        "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
        "achieved_write_gbs_under_decode": round(statistics.mean(eff_gbs), 1),
        "window_ms": round(statistics.mean(win_ms), 2),
        "kernel_blocks": blocks,
        "calibration_gbs_by_blocks": cal,
    }
    print(f"  local_write@{res['achieved_write_gbs_under_decode']}GB/s  "
          f"victim=+{res['victim_slowdown_pct']:.2f}% (±{res['victim_slowdown_std']})  "
          f"window={res['window_ms']:.2f}ms   [g11 peer_in ingress: +23.5% @280]")

    out = {"_experiment": "g11b_ingress_control", "_is_measured": True,
           "_timing_method": "consumer-decode-stream events; single window; clocks locked",
           "device": __import__("torch").cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S, "seeds": args.seeds,
           "note": "locally generated write-only kernel on the g11 victim (cuda:0), "
                   "rate-matched to the ~280 GB/s NVLink-ingress stream of g11 peer_in "
                   "(+23.50/+23.46%). Tests 'ingress writes cost more per byte'.",
           "result": res,
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

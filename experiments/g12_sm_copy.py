"""G12 -- SM-initiated (device-initiated) copies: is direction a knob THERE?

The paper's null covers stream-ordered copy-engine (CE) transfers
(cudaMemcpyPeer-class) -- the path production engines issue. The paper's own
citations document a different world for device-initiated copies: NCCL #512
reports CE writes > reads on A100 NVLink with the preference INVERTING under SM
copies; NVSHMEM measures put 313 vs get 141 GB/s (2.2x). G12 measures that
boundary on the same dual-A100 box with the same controlled protocol:

  sm_push  kernel on HOLDER  (GPU1) reads local HBM, writes remote (GPU0) over NVLink
  sm_pull  kernel on CONSUMER(GPU0) reads remote (GPU1) over NVLink, writes local

The holder decodes throughout (g3 victim protocol: per-iteration events on the
decode stream only, clocks locked, single handoff per seed, no whole-device
sync). Note sm_push burns HOLDER SMs (that is the real cost of SM copies) while
sm_pull burns CONSUMER SMs; both are reported as measured.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g12_sm_copy.json"

CUDA_SRC = r"""
__global__ void copy_kernel(const float4* __restrict__ src,
                            float4* __restrict__ dst, long long n4) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long stride = gridDim.x * (long long)blockDim.x;
  for (; i < n4; i += stride) dst[i] = src[i];
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("enable_peer", &enable_peer);
  m.def("sm_copy", &sm_copy);
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
    gate_or_skip("g12_sm_copy")
    import torch
    import torch.nn.functional as F
    from torch.utils.cpp_extension import load_inline

    ext = load_inline(name="g12_sm_copy_ext", cpp_sources="", cuda_sources=CUDA_SRC,
                      with_cuda=True, verbose=False)
    ext.enable_peer(0, 1)

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = 1, args.ctx
    dt = torch.float16
    # victim decode on the HOLDER (cuda:1), matching g1-g3
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
    nbytes = total * 2
    src_h = torch.randn(total, dtype=dt, device="cuda:1")   # holder-side source
    dst_c = torch.empty(total, dtype=dt, device="cuda:0")   # consumer destination

    dec_stream = torch.cuda.Stream(device=1)
    s_dev1 = torch.cuda.Stream(device=1)   # sm_push: kernel on holder SMs
    s_dev0 = torch.cuda.Stream(device=0)   # sm_pull: kernel on consumer SMs

    def copy_for(kind):
        if kind == "sm_push":   # holder kernel: local read, remote write
            return s_dev1, 1, (lambda: ext.sm_copy(dst_c, src_h, 1, args.blocks))
        if kind == "sm_pull":   # consumer kernel: remote read, local write
            return s_dev0, 0, (lambda: ext.sm_copy(dst_c, src_h, 0, args.blocks))
        raise ValueError(kind)

    for _ in range(30):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    def timed_iter():
        with torch.cuda.device(1), torch.cuda.stream(dec_stream):
            e0.record(dec_stream); decode_step(); e1.record(dec_stream)
        e1.synchronize()
        return e0.elapsed_time(e1)

    def isolated_copy_bw(kind):
        st, dev, op = copy_for(kind)
        ce0 = torch.cuda.Event(enable_timing=True); ce1 = torch.cuda.Event(enable_timing=True)
        for _ in range(3):
            with torch.cuda.device(dev), torch.cuda.stream(st):
                op()
        st.synchronize()
        ts = []
        for _ in range(20):
            with torch.cuda.device(dev), torch.cuda.stream(st):
                ce0.record(st); op(); ce1.record(st)
            ce1.synchronize()
            ts.append(ce0.elapsed_time(ce1))
        return nbytes / (statistics.median(ts) / 1e3) / 1e9

    # correctness check once
    _, dev, op = copy_for("sm_pull")
    with torch.cuda.device(dev), torch.cuda.stream(s_dev0):
        op()
    s_dev0.synchronize()
    assert torch.equal(dst_c.cpu(), src_h.cpu()), "SM copy produced wrong bytes"
    dst_c.zero_()

    results = {}
    for kind in ("sm_push", "sm_pull"):
        bw = isolated_copy_bw(kind)
        slow_seeds, ho_ms = [], []
        for _ in range(args.seeds):
            base = statistics.median([timed_iter() for _ in range(args.base_iters)])
            st, dev, op = copy_for(kind)
            during = []
            t0 = time.time()
            with torch.cuda.device(dev), torch.cuda.stream(st):
                op()
            guard = 0
            while not st.query() and guard < 200000:
                during.append(timed_iter())
                guard += 1
            st.synchronize()
            ho_ms.append((time.time() - t0) * 1e3)
            if during:
                slow_seeds.append(statistics.median(during) / base - 1)
        results[kind] = {
            "copy_bw_gbs": round(bw, 1),
            "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2) if slow_seeds else None,
            "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
            "handoff_ms": round(statistics.mean(ho_ms), 2),
            "kernel_blocks": args.blocks,
        }
        r = results[kind]
        print(f"  {kind:8s} copy_bw={r['copy_bw_gbs']:6.1f}GB/s  victim=+{r['victim_slowdown_pct']:5.2f}% "
              f"(±{r['victim_slowdown_std']})  handoff={r['handoff_ms']:.2f}ms")

    out = {"_experiment": "g12_sm_copy", "_is_measured": True,
           "_timing_method": "decode-stream events; single handoff; clocks locked; no whole-device sync",
           "device": __import__("torch").cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S,
           "handoff_mb": args.handoff_mb, "seeds": args.seeds,
           "note": "SM-initiated copies (grid-stride float4 kernel via P2P). sm_push burns "
                   "holder SMs (local read + remote write); sm_pull burns consumer SMs "
                   "(remote read + local write). CE rows live in g3_placement.json.",
           "results": results,
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

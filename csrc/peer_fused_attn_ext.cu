// peer_fused_attn_ext.cu -- PyTorch CUDA op for multi-device Compute-Follows-KV
// (CFK) attention. This is the op a vLLM custom attention backend calls: it runs a
// split-K flash decode over a LOCAL KV shard (cuda:0) and a PEER KV shard (cuda:1)
// concurrently, pulls the ~KB (O,lse) partial back over NVLink, and merges them
// exactly via online-softmax. Weights and KV never cross devices; only q (broadcast)
// and the peer partial (returned) do.
//
// Python signature (see umallm/peerkv/fused_attn.py):
//   O = peer_fused_attn(q, K_local, V_local, K_peer, V_peer, scale, splits)
//     q       : [H, D]      fp16, cuda:0   (single decode token, batch=1)
//     K/V_loc : [HKV, Tl, D] fp16, cuda:0
//     K/V_peer: [HKV, Tp, D] fp16, cuda:1   (already resident on the peer)
//     returns : [H, D]      fp16, cuda:0
//
// GQA: H query heads, HKV kv heads (H % HKV == 0), head dim D (must be 128 for the
// vectorized warp kernel). Tp may be 0 (degenerates to single-GPU local attention).
//
// Build: via torch.utils.cpp_extension.load() in umallm/peerkv/fused_attn.py, or
// compile into the package extension. Requires the peer devices to have P2P enabled
// (the loader calls cudaDeviceEnablePeerAccess).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "peer_attn_kernels.cuh"

#define TORCH_CK(cond, msg) TORCH_CHECK((cond), msg)

static inline __half* hptr(torch::Tensor& t) { return reinterpret_cast<__half*>(t.data_ptr()); }
static inline const __half* chptr(const torch::Tensor& t) {
    return reinterpret_cast<const __half*>(t.data_ptr());
}

// One-shot multi-device CFK attention. Streams: local work on cuda:0's current
// stream; peer work on a dedicated cuda:1 stream; event fork/join across devices.
torch::Tensor peer_fused_attn(torch::Tensor q,
                              torch::Tensor K_local, torch::Tensor V_local,
                              torch::Tensor K_peer,  torch::Tensor V_peer,
                              double scale, int64_t splits) {
    TORCH_CK(q.scalar_type() == at::kHalf, "q must be fp16");
    TORCH_CK(q.dim() == 2, "q must be [H, D]");
    int H = q.size(0), D = q.size(1);
    TORCH_CK(D == 128, "this kernel requires head_dim D == 128");
    TORCH_CK(K_local.dim() == 3 && V_local.dim() == 3, "K/V_local must be [HKV, Tl, D]");
    int HKV = K_local.size(0), Tl = K_local.size(1);
    TORCH_CK(H % HKV == 0, "H must be a multiple of HKV");
    int dev0 = q.device().index();
    int Tp = (K_peer.defined() && K_peer.numel() > 0) ? (int)K_peer.size(1) : 0;
    int S = (int)splits;
    float sc = (float)scale;

    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    auto f16 = torch::TensorOptions().dtype(torch::kHalf).device(q.device());

    // local scratch on cuda:0
    c10::cuda::CUDAGuard g0(dev0);
    torch::Tensor spO = torch::empty({(long)H*S*D}, f32);
    torch::Tensor spM = torch::empty({(long)H*S}, f32);
    torch::Tensor spL = torch::empty({(long)H*S}, f32);
    torch::Tensor O0  = torch::empty({(long)H*D}, f16);
    torch::Tensor l0  = torch::empty({(long)H}, f32);
    torch::Tensor O   = torch::empty({H, D}, f16);
    cudaStream_t s0 = at::cuda::getCurrentCUDAStream(dev0).stream();

    if (Tp == 0) {
        // single-GPU fast path
        attn_split_gqa<<<H*S, 32, 0, s0>>>(chptr(q), chptr(K_local), chptr(V_local),
            spO.data_ptr<float>(), spM.data_ptr<float>(), spL.data_ptr<float>(),
            H, HKV, Tl, D, S, sc);
        attn_combine<<<H, D, 0, s0>>>(spO.data_ptr<float>(), spM.data_ptr<float>(),
            spL.data_ptr<float>(), hptr(O), l0.data_ptr<float>(), H, D, S);
        return O;
    }

    int dev1 = K_peer.device().index();
    auto f32p = torch::TensorOptions().dtype(torch::kFloat32).device(K_peer.device());
    auto f16p = torch::TensorOptions().dtype(torch::kHalf).device(K_peer.device());

    // peer scratch + q copy on cuda:1
    torch::Tensor qp, spO_p, spM_p, spL_p, O1, l1, O1_recv, l1_recv;
    {
        c10::cuda::CUDAGuard g1(dev1);
        qp     = torch::empty({(long)H*D}, f16p);
        spO_p  = torch::empty({(long)H*S*D}, f32p);
        spM_p  = torch::empty({(long)H*S}, f32p);
        spL_p  = torch::empty({(long)H*S}, f32p);
        O1     = torch::empty({(long)H*D}, f16p);
        l1     = torch::empty({(long)H}, f32p);
    }
    O1_recv = torch::empty({(long)H*D}, f16);
    l1_recv = torch::empty({(long)H}, f32);

    cudaStream_t s1;
    cudaEvent_t e_q, e_peer;
    { c10::cuda::CUDAGuard g1(dev1); cudaStreamCreate(&s1); cudaEventCreate(&e_peer); }
    { c10::cuda::CUDAGuard gg0(dev0); cudaEventCreate(&e_q); }

    // broadcast q to peer, signal
    cudaMemcpyAsync(qp.data_ptr(), q.data_ptr(), (size_t)H*D*sizeof(__half),
                    cudaMemcpyDefault, s0);
    cudaEventRecord(e_q, s0);

    // local attention on cuda:0
    attn_split_gqa<<<H*S, 32, 0, s0>>>(chptr(q), chptr(K_local), chptr(V_local),
        spO.data_ptr<float>(), spM.data_ptr<float>(), spL.data_ptr<float>(),
        H, HKV, Tl, D, S, sc);
    attn_combine<<<H, D, 0, s0>>>(spO.data_ptr<float>(), spM.data_ptr<float>(),
        spL.data_ptr<float>(), hptr(O0), l0.data_ptr<float>(), H, D, S);

    // peer attention on cuda:1 after q arrives
    {
        c10::cuda::CUDAGuard g1(dev1);
        cudaStreamWaitEvent(s1, e_q, 0);
        attn_split_gqa<<<H*S, 32, 0, s1>>>(chptr(qp), chptr(K_peer), chptr(V_peer),
            spO_p.data_ptr<float>(), spM_p.data_ptr<float>(), spL_p.data_ptr<float>(),
            H, HKV, Tp, D, S, sc);
        attn_combine<<<H, D, 0, s1>>>(spO_p.data_ptr<float>(), spM_p.data_ptr<float>(),
            spL_p.data_ptr<float>(), hptr(O1), l1.data_ptr<float>(), H, D, S);
        cudaMemcpyAsync(O1_recv.data_ptr(), O1.data_ptr(), (size_t)H*D*sizeof(__half),
                        cudaMemcpyDefault, s1);
        cudaMemcpyAsync(l1_recv.data_ptr(), l1.data_ptr(), (size_t)H*sizeof(float),
                        cudaMemcpyDefault, s1);
        cudaEventRecord(e_peer, s1);
    }

    // merge on cuda:0 after peer partial arrives
    cudaStreamWaitEvent(s0, e_peer, 0);
    int mB = (H*D + 127) / 128;
    merge2_h<<<mB, 128, 0, s0>>>(hptr(O0), l0.data_ptr<float>(),
        hptr(O1_recv), l1_recv.data_ptr<float>(), hptr(O), H, D);

    // the caller's stream (s0) is ordered after the merge; clean up aux objects
    { c10::cuda::CUDAGuard g1(dev1); cudaStreamSynchronize(s1); cudaStreamDestroy(s1); cudaEventDestroy(e_peer); }
    { c10::cuda::CUDAGuard gg0(dev0); cudaEventDestroy(e_q); }
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("peer_fused_attn", &peer_fused_attn,
          "Multi-device CFK split-K flash attention (GQA). "
          "q[H,D]@cuda0, K/V_local[HKV,Tl,D]@cuda0, K/V_peer[HKV,Tp,D]@cuda1 -> O[H,D]@cuda0",
          py::arg("q"), py::arg("K_local"), py::arg("V_local"),
          py::arg("K_peer"), py::arg("V_peer"), py::arg("scale"), py::arg("splits") = 16);
}

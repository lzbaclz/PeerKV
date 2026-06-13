// peer_fused_decoder.cu  -- Multi-device, fully-fused Compute-Follows-KV (CFK)
// decoder layer (Llama-3-8B GQA geometry by default).
//
// EXTENDS peer_fused_bench.cu (attention-only, measured 1.34-1.98x) to a full layer:
//   RMSNorm -> Wq/Wk/Wv proj (cuBLAS) -> split-K flash attn LOCAL on (K0,V0)
//                                       \-- q broadcast --> peer attn on (K1,V1)
//   merge2 <- (O1,lse1) back over NVLink
//   Wo proj -> residual -> RMSNorm -> Wg/Wu proj -> SwiGLU -> Wd proj -> residual
//
// CFK invariant: WEIGHTS stay whole on cuda:0; only q (~KB) is broadcast to cuda:1
// and only (O1,lse1) (~KB) is pulled back; KV is sharded and never crosses devices.
// The bench step does NOT append (k,v) into the KV cache (KV is treated as already
// resident, consistent with our paged-decode contention model in g1-g4).
//
// Build (standalone bench binary):
//   nvcc -O3 -arch=sm_80 -std=c++14 csrc/peer_fused_decoder.cu \
//        -lcublas -o build/peer_fused_decoder_bench
// Run (defaults Ttot=16384, L=32 layers, trials=40, S=16 splits):
//   ./build/peer_fused_decoder_bench [L] [Ttot] [trials] [S]
//
// Prints per-arm median wall time and reports peer-vs-single ratio and numerics:
//   single | peer-eager | peer-graph
// Numerics: peer (sharded) cosine vs single (full KV) on the final hidden state;
//   should be near 1.0 (online-softmax merge is mathematically exact up to fp order).
//
// Status: complete -- supersedes the prior skeleton. Honest scope:
//   - random fp16 weights/KV (HBM traffic, not values, is what we measure)
//   - one geometry (Llama-3-8B GQA: d=4096, H=32, HKV=8, D=128, DFF=14336)
//   - no KV append, no attention mask, batch=1 decode
//   - cuBLAS workspace pre-allocated to make stream capture deterministic

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include "peer_attn_kernels.cuh"   // attn_split_gqa, attn_combine, merge2_h (shared)

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA err %s @ %s:%d\n",cudaGetErrorString(e),__FILE__,__LINE__);exit(1);} }while(0)
#define BK(x) do{ cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
    printf("cuBLAS err %d @ %s:%d\n",(int)s,__FILE__,__LINE__);exit(1);} }while(0)

// ---------- decoder-only kernels (attention kernels come from the header) ---

// RMSNorm over a [d] vector. One block, threads cooperate on d.
__global__ void rmsnorm_kernel(const __half* __restrict__ x, const __half* __restrict__ w,
                                __half* __restrict__ y, int d, float eps) {
    __shared__ float ss;
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = tid; i < d; i += blockDim.x) { float v = __half2float(x[i]); acc += v*v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    __shared__ float warp_sums[32];
    int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x + 31)/32) ? warp_sums[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) ss = v;
    }
    __syncthreads();
    float rms = rsqrtf(ss / d + eps);
    for (int i = tid; i < d; i += blockDim.x) {
        float v = __half2float(x[i]) * rms * __half2float(w[i]);
        y[i] = __float2half(v);
    }
}

// SwiGLU: y = silu(gate) * up
__global__ void swiglu_kernel(const __half* __restrict__ g, const __half* __restrict__ u,
                              __half* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float gv = __half2float(g[i]); float uv = __half2float(u[i]);
    float s = gv / (1.f + __expf(-gv));
    y[i] = __float2half(s * uv);
}

// Elementwise: y += x
__global__ void add_kernel(__half* __restrict__ y, const __half* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    y[i] = __float2half(__half2float(y[i]) + __half2float(x[i]));
}

// Elementwise: y = x   (used so the captured graph owns the copy)
__global__ void copy_kernel(__half* __restrict__ y, const __half* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    y[i] = x[i];
}

// ---------- gemv (fp16 in/out, fp32 accumulate, cuBLAS) --------------------
// Logical: y[N] = x[K] @ W where W is row-major [K, N]. The same buffer viewed as
// column-major is [N, K]; so use cublasGemmEx with M=N, N=1, K=K, opA=N, opB=N.
static inline void gemv_h(cublasHandle_t h, const __half* W, const __half* x, __half* y,
                          int K, int N, cudaStream_t st) {
    cublasSetStream(h, st);
    const float alpha = 1.f, beta = 0.f;
    BK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K,
                    &alpha, W, CUDA_R_16F, N,
                            x, CUDA_R_16F, K,
                    &beta,  y, CUDA_R_16F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// ---------- types ----------------------------------------------------------

struct LayerWeights {
    __half *rms1, *rms2;           // [d_model]
    __half *Wq, *Wk, *Wv, *Wo;     // Wq[d,H*D] Wk/Wv[d,HKV*D] Wo[H*D,d]
    __half *Wg, *Wu, *Wd;          // Wg/Wu[d,DFF] Wd[DFF,d]
};

struct DecoderConfig {
    int d_model    = 4096;
    int d_ffn      = 14336;
    int n_heads    = 32;
    int n_kv_heads = 8;
    int head_dim   = 128;
    float rms_eps  = 1e-5f;
};

struct WS0 {  // workspace on cuda:0
    __half *x_n, *q, *k, *v, *attn_out, *y1, *y1_n, *gate, *up, *silu_up, *down;
    __half *O0, *O_merged;
    float  *spO, *spM, *spL, *lse0;
    __half *O1_recv;
    float  *l1_recv;
};
struct WS1 {  // workspace on cuda:1
    __half *qp, *O1;
    float  *spO, *spM, *spL, *l1;
};

// ---------- alloc / init ---------------------------------------------------

static __half* dmalloc_h(size_t n) { __half* p; CK(cudaMalloc(&p, n*sizeof(__half))); return p; }
static float*  dmalloc_f(size_t n) { float*  p; CK(cudaMalloc(&p, n*sizeof(float)));  return p; }

static void fill_rand_h(__half* d, size_t n, unsigned seed, float scale) {
    std::vector<__half> h(n);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, scale);
    for (size_t i = 0; i < n; ++i) h[i] = __float2half(nd(rng));
    CK(cudaMemcpy(d, h.data(), n*sizeof(__half), cudaMemcpyHostToDevice));
}

static void alloc_ws0(WS0& w, const DecoderConfig& c, int S) {
    int H = c.n_heads, HKV = c.n_kv_heads, D = c.head_dim;
    w.x_n      = dmalloc_h(c.d_model);
    w.q        = dmalloc_h((size_t)H*D);
    w.k        = dmalloc_h((size_t)HKV*D);
    w.v        = dmalloc_h((size_t)HKV*D);
    w.attn_out = dmalloc_h(c.d_model);
    w.y1       = dmalloc_h(c.d_model);
    w.y1_n     = dmalloc_h(c.d_model);
    w.gate     = dmalloc_h(c.d_ffn);
    w.up       = dmalloc_h(c.d_ffn);
    w.silu_up  = dmalloc_h(c.d_ffn);
    w.down     = dmalloc_h(c.d_model);
    w.O0       = dmalloc_h((size_t)H*D);
    w.O_merged = dmalloc_h((size_t)H*D);
    w.spO      = dmalloc_f((size_t)H*S*D);
    w.spM      = dmalloc_f((size_t)H*S);
    w.spL      = dmalloc_f((size_t)H*S);
    w.lse0     = dmalloc_f(H);
    w.O1_recv  = dmalloc_h((size_t)H*D);
    w.l1_recv  = dmalloc_f(H);
}
static void alloc_ws1(WS1& w, const DecoderConfig& c, int S) {
    int H = c.n_heads, D = c.head_dim;
    w.qp  = dmalloc_h((size_t)H*D);
    w.O1  = dmalloc_h((size_t)H*D);
    w.spO = dmalloc_f((size_t)H*S*D);
    w.spM = dmalloc_f((size_t)H*S);
    w.spL = dmalloc_f((size_t)H*S);
    w.l1  = dmalloc_f(H);
}

// ---------- decoder steps --------------------------------------------------

// PEER: KV sharded as (K0,V0 size Tl on cuda:0) and (K1,V1 size Tp on cuda:1).
static void peer_decoder_layer(const __half* x, __half* y_out,
                                const LayerWeights& W,
                                const __half* K0, const __half* V0, int Tl,
                                const __half* K1, const __half* V1, int Tp,
                                const DecoderConfig& c, int S, float scale,
                                WS0& w0, WS1& w1,
                                cublasHandle_t cu0,
                                cudaStream_t s0, cudaStream_t s1,
                                cudaEvent_t e_q_ready, cudaEvent_t e_peer_done) {
    int H = c.n_heads, HKV = c.n_kv_heads, D = c.head_dim;
    int dm = c.d_model, dff = c.d_ffn;

    CK(cudaSetDevice(0));
    rmsnorm_kernel<<<1, 256, 0, s0>>>(x, W.rms1, w0.x_n, dm, c.rms_eps);
    gemv_h(cu0, W.Wq, w0.x_n, w0.q, dm, H*D,   s0);
    gemv_h(cu0, W.Wk, w0.x_n, w0.k, dm, HKV*D, s0);
    gemv_h(cu0, W.Wv, w0.x_n, w0.v, dm, HKV*D, s0);
    // Broadcast q to cuda:1 (overlaps with local attn). NB: k,v are computed locally and
    // would normally be appended to the KV cache; we do not model the append step here.
    CK(cudaMemcpyAsync(w1.qp, w0.q, (size_t)H*D*sizeof(__half), cudaMemcpyDefault, s0));
    CK(cudaEventRecord(e_q_ready, s0));

    attn_split_gqa<<<H*S, 32, 0, s0>>>(w0.q, K0, V0, w0.spO, w0.spM, w0.spL,
                                       H, HKV, Tl, D, S, scale);
    attn_combine<<<H, D, 0, s0>>>(w0.spO, w0.spM, w0.spL, w0.O0, w0.lse0, H, D, S);

    CK(cudaSetDevice(1));
    CK(cudaStreamWaitEvent(s1, e_q_ready, 0));
    attn_split_gqa<<<H*S, 32, 0, s1>>>(w1.qp, K1, V1, w1.spO, w1.spM, w1.spL,
                                       H, HKV, Tp, D, S, scale);
    attn_combine<<<H, D, 0, s1>>>(w1.spO, w1.spM, w1.spL, w1.O1, w1.l1, H, D, S);
    CK(cudaMemcpyAsync(w0.O1_recv, w1.O1, (size_t)H*D*sizeof(__half), cudaMemcpyDefault, s1));
    CK(cudaMemcpyAsync(w0.l1_recv, w1.l1, (size_t)H*sizeof(float),    cudaMemcpyDefault, s1));
    CK(cudaEventRecord(e_peer_done, s1));

    CK(cudaSetDevice(0));
    CK(cudaStreamWaitEvent(s0, e_peer_done, 0));
    int mB = (H*D + 127) / 128;
    merge2_h<<<mB, 128, 0, s0>>>(w0.O0, w0.lse0, w0.O1_recv, w0.l1_recv, w0.O_merged, H, D);

    gemv_h(cu0, W.Wo, w0.O_merged, w0.attn_out, H*D, dm, s0);
    copy_kernel<<<(dm + 255)/256, 256, 0, s0>>>(w0.y1, x, dm);
    add_kernel <<<(dm + 255)/256, 256, 0, s0>>>(w0.y1, w0.attn_out, dm);

    rmsnorm_kernel<<<1, 256, 0, s0>>>(w0.y1, W.rms2, w0.y1_n, dm, c.rms_eps);
    gemv_h(cu0, W.Wg, w0.y1_n, w0.gate, dm,  dff, s0);
    gemv_h(cu0, W.Wu, w0.y1_n, w0.up,   dm,  dff, s0);
    swiglu_kernel<<<(dff + 255)/256, 256, 0, s0>>>(w0.gate, w0.up, w0.silu_up, dff);
    gemv_h(cu0, W.Wd, w0.silu_up, w0.down, dff, dm, s0);
    copy_kernel<<<(dm + 255)/256, 256, 0, s0>>>(y_out, w0.y1, dm);
    add_kernel <<<(dm + 255)/256, 256, 0, s0>>>(y_out, w0.down, dm);
}

// SINGLE: full KV (Ttot) resident on cuda:0; no peer.
static void single_decoder_layer(const __half* x, __half* y_out,
                                  const LayerWeights& W,
                                  const __half* Kf, const __half* Vf, int Ttot,
                                  const DecoderConfig& c, int S, float scale,
                                  WS0& w0, cublasHandle_t cu0, cudaStream_t s0) {
    int H = c.n_heads, HKV = c.n_kv_heads, D = c.head_dim;
    int dm = c.d_model, dff = c.d_ffn;
    CK(cudaSetDevice(0));
    rmsnorm_kernel<<<1, 256, 0, s0>>>(x, W.rms1, w0.x_n, dm, c.rms_eps);
    gemv_h(cu0, W.Wq, w0.x_n, w0.q, dm, H*D,   s0);
    gemv_h(cu0, W.Wk, w0.x_n, w0.k, dm, HKV*D, s0);
    gemv_h(cu0, W.Wv, w0.x_n, w0.v, dm, HKV*D, s0);
    attn_split_gqa<<<H*S, 32, 0, s0>>>(w0.q, Kf, Vf, w0.spO, w0.spM, w0.spL,
                                       H, HKV, Ttot, D, S, scale);
    attn_combine<<<H, D, 0, s0>>>(w0.spO, w0.spM, w0.spL, w0.O0, w0.lse0, H, D, S);
    gemv_h(cu0, W.Wo, w0.O0, w0.attn_out, H*D, dm, s0);
    copy_kernel<<<(dm + 255)/256, 256, 0, s0>>>(w0.y1, x, dm);
    add_kernel <<<(dm + 255)/256, 256, 0, s0>>>(w0.y1, w0.attn_out, dm);
    rmsnorm_kernel<<<1, 256, 0, s0>>>(w0.y1, W.rms2, w0.y1_n, dm, c.rms_eps);
    gemv_h(cu0, W.Wg, w0.y1_n, w0.gate, dm,  dff, s0);
    gemv_h(cu0, W.Wu, w0.y1_n, w0.up,   dm,  dff, s0);
    swiglu_kernel<<<(dff + 255)/256, 256, 0, s0>>>(w0.gate, w0.up, w0.silu_up, dff);
    gemv_h(cu0, W.Wd, w0.silu_up, w0.down, dff, dm, s0);
    copy_kernel<<<(dm + 255)/256, 256, 0, s0>>>(y_out, w0.y1, dm);
    add_kernel <<<(dm + 255)/256, 256, 0, s0>>>(y_out, w0.down, dm);
}

// ---------- C entry point (for Python bindings; bench main is below) ------

extern "C" int peer_fused_decoder_step(/* opaque runner */ void* /*r*/) {
    // Not exposed in the standalone bench (main() drives everything); kept so the
    // symbol exists for future ctypes/cffi bindings from Track C runtime.
    return 0;
}

// ---------- bench main -----------------------------------------------------

static double bench_median(std::vector<double> ts) {
    std::sort(ts.begin(), ts.end()); return ts[ts.size()/2];
}

static double host_cosine(const __half* da, const __half* db, int n) {
    std::vector<__half> a(n), b(n);
    CK(cudaMemcpy(a.data(), da, n*sizeof(__half), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), db, n*sizeof(__half), cudaMemcpyDeviceToHost));
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        double xa = __half2float(a[i]), xb = __half2float(b[i]);
        dot += xa*xb; na += xa*xa; nb += xb*xb;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

int main(int argc, char** argv) {
    int L = 32, Ttot = 16384, trials = 40, S = 16;
    if (argc > 1) L      = atoi(argv[1]);
    if (argc > 2) Ttot   = atoi(argv[2]);
    if (argc > 3) trials = atoi(argv[3]);
    if (argc > 4) S      = atoi(argv[4]);

    DecoderConfig c;  // Llama-3-8B GQA defaults
    int H = c.n_heads, HKV = c.n_kv_heads, D = c.head_dim;
    int dm = c.d_model, dff = c.d_ffn;
    int Tl = Ttot/2, Tp = Ttot - Tl;
    float scale = 1.f / std::sqrt((float)D);

    int nd; CK(cudaGetDeviceCount(&nd));
    if (nd < 2) { printf("need 2 GPUs (have %d)\n", nd); return 1; }
    CK(cudaSetDevice(0)); cudaDeviceEnablePeerAccess(1, 0);
    CK(cudaSetDevice(1)); cudaDeviceEnablePeerAccess(0, 0);

    // Per-layer weights on cuda:0; KV sharded.
    std::vector<LayerWeights> Wv(L);
    std::vector<__half*> Kf(L), Vf(L), K0(L), V0(L), K1(L), V1(L);
    size_t kvF = (size_t)HKV*Ttot*D, kv0 = (size_t)HKV*Tl*D, kv1 = (size_t)HKV*Tp*D;

    CK(cudaSetDevice(0));
    for (int i = 0; i < L; ++i) {
        Wv[i].rms1 = dmalloc_h(dm);   fill_rand_h(Wv[i].rms1, dm, 1+i*7, 1.0f);
        Wv[i].rms2 = dmalloc_h(dm);   fill_rand_h(Wv[i].rms2, dm, 2+i*7, 1.0f);
        Wv[i].Wq   = dmalloc_h((size_t)dm*H*D);   fill_rand_h(Wv[i].Wq, (size_t)dm*H*D, 3+i*7, 0.02f);
        Wv[i].Wk   = dmalloc_h((size_t)dm*HKV*D); fill_rand_h(Wv[i].Wk, (size_t)dm*HKV*D, 4+i*7, 0.02f);
        Wv[i].Wv   = dmalloc_h((size_t)dm*HKV*D); fill_rand_h(Wv[i].Wv, (size_t)dm*HKV*D, 5+i*7, 0.02f);
        Wv[i].Wo   = dmalloc_h((size_t)H*D*dm);   fill_rand_h(Wv[i].Wo, (size_t)H*D*dm,   6+i*7, 0.02f);
        Wv[i].Wg   = dmalloc_h((size_t)dm*dff);   fill_rand_h(Wv[i].Wg, (size_t)dm*dff,   7+i*7, 0.02f);
        Wv[i].Wu   = dmalloc_h((size_t)dm*dff);   fill_rand_h(Wv[i].Wu, (size_t)dm*dff,   8+i*7, 0.02f);
        Wv[i].Wd   = dmalloc_h((size_t)dff*dm);   fill_rand_h(Wv[i].Wd, (size_t)dff*dm,   9+i*7, 0.02f);
        Kf[i] = dmalloc_h(kvF); fill_rand_h(Kf[i], kvF, 11+i*5, 0.02f);
        Vf[i] = dmalloc_h(kvF); fill_rand_h(Vf[i], kvF, 12+i*5, 0.02f);
        K0[i] = dmalloc_h(kv0); V0[i] = dmalloc_h(kv0);
        // Per-head shard: K0 holds tokens [0..Tl) of every kv-head; the K layout is
        // [HKV, T, D] row-major, so the shards must be filled per-head, not byte-contiguous.
        for (int k = 0; k < HKV; ++k) {
            CK(cudaMemcpy(K0[i] + (size_t)k*Tl*D, Kf[i] + (size_t)k*Ttot*D,
                          (size_t)Tl*D*sizeof(__half), cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(V0[i] + (size_t)k*Tl*D, Vf[i] + (size_t)k*Ttot*D,
                          (size_t)Tl*D*sizeof(__half), cudaMemcpyDeviceToDevice));
        }
    }
    // K1/V1 = the tail [Tl..Ttot) tokens of each kv-head, hosted on cuda:1.
    CK(cudaSetDevice(1));
    for (int i = 0; i < L; ++i) {
        K1[i] = dmalloc_h(kv1); V1[i] = dmalloc_h(kv1);
        for (int k = 0; k < HKV; ++k) {
            CK(cudaMemcpy(K1[i] + (size_t)k*Tp*D, Kf[i] + (size_t)k*Ttot*D + (size_t)Tl*D,
                          (size_t)Tp*D*sizeof(__half), cudaMemcpyDefault));
            CK(cudaMemcpy(V1[i] + (size_t)k*Tp*D, Vf[i] + (size_t)k*Ttot*D + (size_t)Tl*D,
                          (size_t)Tp*D*sizeof(__half), cudaMemcpyDefault));
        }
    }

    // Workspaces, streams, events, cuBLAS.
    WS0 w0; WS1 w1;
    CK(cudaSetDevice(0)); alloc_ws0(w0, c, S);
    CK(cudaSetDevice(1)); alloc_ws1(w1, c, S);

    cudaStream_t s0, s1;
    CK(cudaSetDevice(0)); CK(cudaStreamCreate(&s0));
    CK(cudaSetDevice(1)); CK(cudaStreamCreate(&s1));
    cudaEvent_t e_q, e_p, t_a, t_b;
    CK(cudaSetDevice(0)); CK(cudaEventCreate(&e_q)); CK(cudaEventCreate(&t_a)); CK(cudaEventCreate(&t_b));
    CK(cudaSetDevice(1)); CK(cudaEventCreate(&e_p));

    cublasHandle_t cu0; CK(cudaSetDevice(0)); BK(cublasCreate(&cu0));
    BK(cublasSetMathMode(cu0, CUBLAS_TENSOR_OP_MATH));
    void* cu_ws; size_t cu_ws_bytes = 32ull * 1024 * 1024;
    CK(cudaMalloc(&cu_ws, cu_ws_bytes));
    BK(cublasSetWorkspace(cu0, cu_ws, cu_ws_bytes));

    // The hidden state: ping-pong x_buf <-> y_buf across layers.
    __half *x_buf, *y_buf; CK(cudaSetDevice(0));
    x_buf = dmalloc_h(dm); fill_rand_h(x_buf, dm, 99, 0.02f);
    y_buf = dmalloc_h(dm);

    auto run_single = [&]{
        __half* xp = x_buf; __half* yp = y_buf;
        for (int i = 0; i < L; ++i) {
            single_decoder_layer(xp, yp, Wv[i], Kf[i], Vf[i], Ttot, c, S, scale, w0, cu0, s0);
            std::swap(xp, yp);
        }
    };
    auto run_peer = [&]{
        __half* xp = x_buf; __half* yp = y_buf;
        for (int i = 0; i < L; ++i) {
            peer_decoder_layer(xp, yp, Wv[i], K0[i], V0[i], Tl, K1[i], V1[i], Tp,
                               c, S, scale, w0, w1, cu0, s0, s1, e_q, e_p);
            std::swap(xp, yp);
        }
    };

    auto sync_both = [&]{ CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
                           CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize()); };

    // warm
    for (int w = 0; w < 4; ++w) { run_single(); run_peer(); }
    sync_both();

    // ---- numerics check: single vs peer on the same x, weights, KV ----
    // Critical: warmup above corrupts x_buf/y_buf. Reset x_buf to the SAME initial
    // seed for both arms; otherwise we compare outputs of two different inputs.
    fill_rand_h(x_buf, dm, 99, 0.02f);
    run_single(); sync_both();
    __half* single_out = dmalloc_h(dm);
    __half* fin_single = (L % 2 == 0) ? x_buf : y_buf;
    CK(cudaMemcpy(single_out, fin_single, dm*sizeof(__half), cudaMemcpyDeviceToDevice));
    fill_rand_h(x_buf, dm, 99, 0.02f);   // same starting state for peer
    run_peer(); sync_both();
    __half* fin_peer = (L % 2 == 0) ? x_buf : y_buf;
    double cos = host_cosine(single_out, fin_peer, dm);
    fill_rand_h(x_buf, dm, 99, 0.02f);  // reset for timing
    CK(cudaFree(single_out));

    // ---- timing ----
    auto time_arm = [&](auto fn) {
        sync_both();
        std::vector<double> ts;
        for (int r = 0; r < trials; ++r) {
            sync_both();
            CK(cudaSetDevice(0)); CK(cudaEventRecord(t_a, s0));
            fn();
            CK(cudaEventRecord(t_b, s0));
            sync_both();
            float ms; CK(cudaEventElapsedTime(&ms, t_a, t_b)); ts.push_back(ms);
        }
        return bench_median(ts);
    };

    double sg = time_arm(run_single);
    double pe = time_arm(run_peer);

    // ---- try multi-device graph capture for peer (mirrors peer_fused_bench.cu) ----
    bool graph_ok = true; cudaGraph_t g{}; cudaGraphExec_t ge{};
    double pg = -1.0;
    CK(cudaSetDevice(0));
    if (cudaStreamBeginCapture(s0, cudaStreamCaptureModeGlobal) != cudaSuccess) graph_ok = false;
    if (graph_ok) {
        run_peer();
        if (cudaStreamEndCapture(s0, &g) != cudaSuccess) graph_ok = false;
    }
    if (graph_ok && cudaGraphInstantiate(&ge, g, 0) != cudaSuccess) graph_ok = false;
    if (graph_ok) {
        for (int w = 0; w < 4; ++w) CK(cudaGraphLaunch(ge, s0));
        sync_both();
        std::vector<double> ts;
        for (int r = 0; r < trials; ++r) {
            sync_both();
            CK(cudaEventRecord(t_a, s0));
            CK(cudaGraphLaunch(ge, s0));
            CK(cudaEventRecord(t_b, s0));
            sync_both();
            float ms; CK(cudaEventElapsedTime(&ms, t_a, t_b)); ts.push_back(ms);
        }
        pg = bench_median(ts);
    }

    printf("# Llama-3-8B GQA fused decoder bench\n");
    printf("# H=%d HKV=%d D=%d dmodel=%d dffn=%d  L=%d Ttot=%d (Tl=%d Tp=%d) S=%d trials=%d\n",
           H, HKV, D, dm, dff, L, Ttot, Tl, Tp, S, trials);
    printf("# numerics: cosine(single, peer) = %.6f  (merge is exact up to fp order)\n", cos);
    printf("single      %8.3f ms  (%7.2f us/layer)\n", sg, sg/L*1000.0);
    printf("peer-eager  %8.3f ms  ratio single/peer = %.3fx%s\n",
           pe, sg/pe, (sg/pe >= 1.0 ? "  (>=1: peer matches/beats single)" : ""));
    if (graph_ok)
        printf("peer-graph  %8.3f ms  ratio single/peer = %.3fx%s\n",
               pg, sg/pg, (sg/pg >= 1.0 ? "  (>=1: peer matches/beats single)" : ""));
    else
        printf("peer-graph  -- capture failed (cross-device cuBLAS in captured graph; eager arm still valid)\n");
    return 0;
}

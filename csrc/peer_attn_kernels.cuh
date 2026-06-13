// peer_attn_kernels.cuh -- shared split-K GQA flash-decode kernels for the
// multi-device Compute-Follows-KV (CFK) path. Single source of truth used by:
//   - csrc/peer_fused_decoder.cu        (standalone full-layer bench)
//   - csrc/peer_fused_attn_ext.cu       (PyTorch CUDA op for the vLLM backend)
//
// Layout: Q [H, D]; K,V contiguous per kv-head [HKV, T, D] (row-major). GQA: query
// head h reads kv-head h/(H/HKV). Partials are online-softmax (m,l,O) so two device
// shards merge exactly via merge2_h.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// phase 1: one warp per (q_head, split). Each lane owns 4 dims (2x half2).
__global__ void attn_split_gqa(const __half* __restrict__ q,
                               const __half* __restrict__ K, const __half* __restrict__ V,
                               float* __restrict__ Op, float* __restrict__ mp, float* __restrict__ lp,
                               int H, int HKV, int T, int D, int S, float scale) {
    int blk = blockIdx.x, s = blk % S, h = blk / S, lane = threadIdx.x;
    int groups = H / HKV; int kv_head = h / groups;
    int chunk = (T + S - 1) / S, t0 = s * chunk, t1 = min(T, t0 + chunk);
    const __half2* q2 = (const __half2*)(q + (size_t)h * D);
    __half2 qa = q2[lane*2], qb = q2[lane*2 + 1];
    float m = -1e30f, l = 0.f, a0=0, a1=0, a2=0, a3=0;
    const __half* Kh = K + (size_t)kv_head * T * D;
    const __half* Vh = V + (size_t)kv_head * T * D;
    if (t0 >= t1) {
        int idx = h*S + s; float* op = Op + (size_t)idx*D + lane*4;
        op[0]=0; op[1]=0; op[2]=0; op[3]=0;
        if (lane == 0) { mp[idx] = m; lp[idx] = 0.f; }
        return;
    }
    for (int t = t0; t < t1; ++t) {
        const __half2* k2 = (const __half2*)(Kh + (size_t)t * D);
        __half2 pr = __hadd2(__hmul2(qa, k2[lane*2]), __hmul2(qb, k2[lane*2 + 1]));
        float p = __low2float(pr) + __high2float(pr);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
        p = __shfl_sync(0xffffffffu, p, 0) * scale;
        float mn = fmaxf(m, p), c = __expf(m - mn), pe = __expf(p - mn); l = l*c + pe;
        const __half2* v2 = (const __half2*)(Vh + (size_t)t * D);
        __half2 va = v2[lane*2], vb = v2[lane*2 + 1];
        a0 = a0*c + pe*__low2float(va); a1 = a1*c + pe*__high2float(va);
        a2 = a2*c + pe*__low2float(vb); a3 = a3*c + pe*__high2float(vb); m = mn;
    }
    int idx = h*S + s; float* op = Op + (size_t)idx*D + lane*4;
    op[0]=a0; op[1]=a1; op[2]=a2; op[3]=a3; if (lane == 0) { mp[idx] = m; lp[idx] = l; }
}

// phase 2: one block per head; merge S partials -> (O[H,D] fp16, lse[H] fp32).
__global__ void attn_combine(const float* Op, const float* mp, const float* lp,
                             __half* O, float* lse, int H, int D, int S) {
    int h = blockIdx.x, d = threadIdx.x;
    float m = -1e30f, l = 0.f, acc = 0.f;
    for (int s = 0; s < S; ++s) {
        int idx = h*S + s; float ms = mp[idx], ls = lp[idx], os = Op[(size_t)idx*D + d];
        float mn = fmaxf(m, ms), c = __expf(m - mn), cs = __expf(ms - mn);
        l = l*c + ls*cs; acc = acc*c + cs*os; m = mn;
    }
    O[h*D + d] = __float2half(acc / l);
    if (d == 0) lse[h] = m + logf(l);
}

// merge two device-partial (O,lse) into one O[H,D].
__global__ void merge2_h(const __half* O0, const float* l0,
                         const __half* O1, const float* l1,
                         __half* O, int H, int D) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= H*D) return;
    int h = i / D;
    float a = l0[h], b = l1[h], mx = fmaxf(a, b);
    float w0 = __expf(a - mx), w1 = __expf(b - mx);
    float o = (__half2float(O0[i])*w0 + __half2float(O1[i])*w1) / (w0 + w1);
    O[i] = __float2half(o);
}

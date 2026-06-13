// peer_rt_bench.cu -- proof that hand-written CUDA collapses the per-layer
// cross-GPU round-trip that PyTorch cannot.
//
// Same decode-attention + online-softmax-merge kernels, run two ways over L
// layers (KV sharded: half the context resident on GPU0, half on GPU1):
//   (A) EAGER : raw per-layer launches + cudaMemcpyPeerAsync (mirrors PyTorch eager)
//   (B) GRAPH : the ENTIRE L-layer cross-device sequence captured into ONE
//               multi-device cudaGraph (cudaStreamBeginCapture + cross-device
//               event fork/join + cudaMemcpyPeerAsync nodes), replayed with one
//               cudaGraphLaunch per token.
// (B) is exactly what torch.cuda.graph cannot do (single-device). We also verify
// the graph output equals the eager output (correctness), then report per-layer
// round-trip time for each. Build: nvcc -O3 -arch=sm_80.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)

// one block per head; blockDim = D threads. Online-softmax decode attention:
// q[H,D], K/V[H,T,D] (half) -> O[H,D] (float), lse[H] (float).
__global__ void decode_attn(const __half* __restrict__ q, const __half* __restrict__ K,
                            const __half* __restrict__ V, float* __restrict__ O,
                            float* __restrict__ lse, int H, int T, int D, float scale){
  int h = blockIdx.x, d = threadIdx.x;        // D == blockDim.x
  if(h>=H) return;
  extern __shared__ float sh[];               // [D] for dot reduction
  float qd = __half2float(q[h*D+d]);
  float m=-1e30f, l=0.f, acc=0.f;
  const __half* Kh = K + (size_t)h*T*D;
  const __half* Vh = V + (size_t)h*T*D;
  for(int t=0;t<T;++t){
    float prod = qd * __half2float(Kh[(size_t)t*D+d]);
    sh[d]=prod; __syncthreads();
    for(int s=D>>1;s>0;s>>=1){ if(d<s) sh[d]+=sh[d+s]; __syncthreads(); }
    float score = sh[0]*scale; __syncthreads();
    float m_new = fmaxf(m, score);
    float corr = __expf(m - m_new);
    float p = __expf(score - m_new);
    l = l*corr + p;
    acc = acc*corr + p*__half2float(Vh[(size_t)t*D+d]);
    m = m_new;
  }
  O[h*D+d] = acc / l;
  if(d==0) lse[h] = m + logf(l);
}

// merge two (O,lse) partials -> O (float). H*D threads.
__global__ void merge_kernel(const float* O0,const float* lse0,const float* O1,
                             const float* lse1,float* O,int H,int D){
  int i = blockIdx.x*blockDim.x + threadIdx.x; if(i>=H*D) return;
  int h = i / D;
  float a=lse0[h], b=lse1[h], mx=fmaxf(a,b);
  float w0=__expf(a-mx), w1=__expf(b-mx), denom=w0+w1;
  O[i] = (O0[i]*w0 + O1[i]*w1)/denom;
}

int main(int argc,char**argv){
  int L=32, H=32, D=128, Ttot=16384, trials=50;
  if(argc>1) L=atoi(argv[1]); if(argc>2) Ttot=atoi(argv[2]); if(argc>3) trials=atoi(argv[3]);
  int Tl=Ttot/2, Tp=Ttot-Tl; float scale=1.f/sqrtf((float)D);
  int ndev=0; CK(cudaGetDeviceCount(&ndev)); if(ndev<2){printf("need 2 GPUs\n");return 1;}
  // enable peer access both ways
  CK(cudaSetDevice(0)); cudaDeviceEnablePeerAccess(1,0);
  CK(cudaSetDevice(1)); cudaDeviceEnablePeerAccess(0,0);

  size_t kv0 = (size_t)H*Tl*D, kv1=(size_t)H*Tp*D, od=(size_t)H*D;
  std::vector<__half*> K0(L),V0(L),K1(L),V1(L); std::vector<__half*> qd(L);
  std::vector<float*> O0(L),l0(L),O1(L),l1(L),O1h(L),l1h(L),Oout(L); __half* qpeer;
  // alloc: home(0) tensors + a peer q buffer; peer(1) KV + peer partials
  CK(cudaSetDevice(0));
  for(int i=0;i<L;++i){ CK(cudaMalloc(&K0[i],kv0*2)); CK(cudaMalloc(&V0[i],kv0*2));
    CK(cudaMalloc(&qd[i],od*2)); CK(cudaMalloc(&O0[i],od*4)); CK(cudaMalloc(&l0[i],H*4));
    CK(cudaMalloc(&O1h[i],od*4)); CK(cudaMalloc(&l1h[i],H*4)); CK(cudaMalloc(&Oout[i],od*4)); }
  CK(cudaMalloc(&qpeer,od*2));
  CK(cudaSetDevice(1));
  for(int i=0;i<L;++i){ CK(cudaMalloc(&K1[i],kv1*2)); CK(cudaMalloc(&V1[i],kv1*2));
    CK(cudaMalloc(&O1[i],od*4)); CK(cudaMalloc(&l1[i],H*4)); }
  // init with small values so softmax is well-conditioned
  CK(cudaSetDevice(0)); for(int i=0;i<L;++i){ CK(cudaMemset(K0[i],0,kv0*2)); CK(cudaMemset(V0[i],0,kv0*2)); CK(cudaMemset(qd[i],0,od*2)); }
  CK(cudaSetDevice(1)); for(int i=0;i<L;++i){ CK(cudaMemset(K1[i],0,kv1*2)); CK(cudaMemset(V1[i],0,kv1*2)); }
  CK(cudaDeviceSynchronize()); CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());

  cudaStream_t s0,s1; CK(cudaSetDevice(0)); CK(cudaStreamCreate(&s0));
  CK(cudaSetDevice(1)); CK(cudaStreamCreate(&s1));
  // an event must be RECORDED on a stream of the device it was created on;
  // e_h2p is signalled on s0 (dev0), e_p2h on s1 (dev1). Cross-device WAIT is fine.
  cudaEvent_t e_h2p, e_p2h;
  CK(cudaSetDevice(0)); CK(cudaEventCreate(&e_h2p));
  CK(cudaSetDevice(1)); CK(cudaEventCreate(&e_p2h));
  int mergeBlocks=(H*D+127)/128; size_t shmem=D*sizeof(float);

  // one layer's cross-device work issued onto streams s0(home)/s1(peer)
  auto issue_layer=[&](int i){
    // home(0): copy q -> peer, signal e_h2p (each stream op under its own device)
    CK(cudaSetDevice(0));
    CK(cudaMemcpyAsync(qpeer,qd[i],od*2,cudaMemcpyDefault,s0));  // capturable x-dev copy
    CK(cudaEventRecord(e_h2p,s0));
    // peer(1): wait q, flash over peer KV, copy partial -> home, signal e_p2h
    CK(cudaSetDevice(1));
    CK(cudaStreamWaitEvent(s1,e_h2p,0));
    decode_attn<<<H,D,shmem,s1>>>(qpeer,K1[i],V1[i],O1[i],l1[i],H,Tp,D,scale);
    CK(cudaMemcpyAsync(O1h[i],O1[i],od*4,cudaMemcpyDefault,s1));
    CK(cudaMemcpyAsync(l1h[i],l1[i],H*4,cudaMemcpyDefault,s1));
    CK(cudaEventRecord(e_p2h,s1));
    // home(0): local flash CONCURRENTLY, then wait peer partial, then merge
    CK(cudaSetDevice(0));
    decode_attn<<<H,D,shmem,s0>>>(qd[i],K0[i],V0[i],O0[i],l0[i],H,Tl,D,scale);
    CK(cudaStreamWaitEvent(s0,e_p2h,0));
    merge_kernel<<<mergeBlocks,128,0,s0>>>(O0[i],l0[i],O1h[i],l1h[i],Oout[i],H,D);
  };

  // ---- EAGER: per-layer raw launches, timed ----
  CK(cudaSetDevice(0));
  for(int w=0;w<5;++w){ for(int i=0;i<L;++i) issue_layer(i); }
  CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
  cudaEvent_t t0,t1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  std::vector<float> eager;
  for(int r=0;r<trials;++r){ CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize());
    CK(cudaSetDevice(0)); CK(cudaEventRecord(t0,s0));
    for(int i=0;i<L;++i) issue_layer(i);
    CK(cudaEventRecord(t1,s0)); CK(cudaSetDevice(1)); CK(cudaDeviceSynchronize()); CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
    float ms; CK(cudaEventElapsedTime(&ms,t0,t1)); eager.push_back(ms); }
  std::sort(eager.begin(),eager.end()); float eg=eager[eager.size()/2];

  // save eager output (layer 0) for correctness
  std::vector<float> ref(od); CK(cudaMemcpy(ref.data(),Oout[0],od*4,cudaMemcpyDeviceToHost));

  // ---- GRAPH: capture the whole L-layer cross-device sequence into ONE graph ----
  cudaGraph_t graph; cudaGraphExec_t gexec; bool graph_ok=true; const char* gerr="";
  CK(cudaSetDevice(0));
  cudaError_t cap=cudaStreamBeginCapture(s0,cudaStreamCaptureModeGlobal);
  if(cap!=cudaSuccess){graph_ok=false;gerr=cudaGetErrorString(cap);}
  if(graph_ok){ for(int i=0;i<L;++i) issue_layer(i);
    cudaError_t ec=cudaStreamEndCapture(s0,&graph);
    if(ec!=cudaSuccess){graph_ok=false;gerr=cudaGetErrorString(ec);} }
  float gg=-1; std::vector<float> gout(od);
  if(graph_ok){
    cudaError_t ie=cudaGraphInstantiate(&gexec,graph,0);
    if(ie!=cudaSuccess){graph_ok=false;gerr=cudaGetErrorString(ie);} }
  if(graph_ok){
    for(int w=0;w<5;++w) CK(cudaGraphLaunch(gexec,s0));
    CK(cudaSetDevice(0)); CK(cudaDeviceSynchronize());
    std::vector<float> gt;
    for(int r=0;r<trials;++r){ CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(t0,s0)); CK(cudaGraphLaunch(gexec,s0)); CK(cudaEventRecord(t1,s0));
      CK(cudaDeviceSynchronize()); float ms; CK(cudaEventElapsedTime(&ms,t0,t1)); gt.push_back(ms); }
    std::sort(gt.begin(),gt.end()); gg=gt[gt.size()/2];
    CK(cudaMemcpy(gout.data(),Oout[0],od*4,cudaMemcpyDeviceToHost));
  }

  double maxerr=0; for(size_t i=0;i<od;++i) maxerr=std::max(maxerr,(double)fabs(ref[i]-gout[i]));
  printf("config: L=%d Ttot=%d (home %d / peer %d) H=%d D=%d trials=%d\n",L,Ttot,Tl,Tp,H,D,trials);
  printf("EAGER (raw per-layer)     : %.3f ms  (%.1f us/layer)\n", eg, eg/L*1000); fflush(stdout);
  if(graph_ok){
    printf("GRAPH (1 multi-dev graph) : %.3f ms  (%.1f us/layer)  %.2fx vs eager\n", gg, gg/L*1000, eg/gg);
    printf("correctness max|graph-eager| = %.3e (expect ~0)\n", maxerr);
  } else printf("GRAPH capture FAILED: %s\n", gerr);
  return 0;
}

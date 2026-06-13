// peer_fused_bench.cu -- does peer-parallel attention-ONLY reach <= single-GPU
// once the decode kernel is FAST (split-K, fills the SMs -> bandwidth-bound)?
//
// Split-K flash decode: phase-1 grid (H*S) computes per-(head,split) online-softmax
// partials over a T-chunk (warp-shuffle dot, coalesced KV); phase-2 (H) merges the S
// partials. This fills the GPU so the kernel is bandwidth-bound (realistic), unlike a
// one-block-per-head kernel. Same launcher for all arms:
//   single : full KV (Ttot) on GPU0
//   peer-E : sharded (Tl/Tp), EAGER cross-device per layer
//   peer-G : sharded, ONE multi-device CUDA graph
// peer <= single iff (half-KV read) >= (cross-device round-trip). Build: nvcc -O3 -arch=sm_80.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s @ %d\n",cudaGetErrorString(e),__LINE__);exit(1);} }while(0)

// phase 1: WARP per (head, split) -- blockDim=32, D=128. Each lane owns 4 dims via
// 2x half2 (vectorized 64-bit coalesced loads of K/V); warp-shuffle dot reduction,
// NO __syncthreads (single warp). Bandwidth-bound. Partial -> (mp,lp,Op[D]) @ h*S+s.
__global__ void attn_split(const __half* __restrict__ q,const __half* __restrict__ K,
                           const __half* __restrict__ V,float* Op,float* mp,float* lp,
                           int H,int T,int D,int S,float scale){
  int blk=blockIdx.x, s=blk%S, h=blk/S, lane=threadIdx.x;   // 32 lanes
  int chunk=(T+S-1)/S, t0=s*chunk, t1=min(T,t0+chunk);
  const __half2* q2=(const __half2*)(q+(size_t)h*D);
  __half2 qa=q2[lane*2], qb=q2[lane*2+1];                   // lane owns dims [4*lane..+3]
  float m=-1e30f,l=0.f,a0=0,a1=0,a2=0,a3=0;
  const __half* Kh=K+(size_t)h*T*D; const __half* Vh=V+(size_t)h*T*D;
  for(int t=t0;t<t1;++t){
    const __half2* k2=(const __half2*)(Kh+(size_t)t*D);
    __half2 pr=__hadd2(__hmul2(qa,k2[lane*2]),__hmul2(qb,k2[lane*2+1]));
    float p=__low2float(pr)+__high2float(pr);
    #pragma unroll
    for(int o=16;o>0;o>>=1) p+=__shfl_down_sync(0xffffffffu,p,o);
    p=__shfl_sync(0xffffffffu,p,0)*scale;                   // broadcast score
    float mn=fmaxf(m,p),c=__expf(m-mn),pe=__expf(p-mn); l=l*c+pe;
    const __half2* v2=(const __half2*)(Vh+(size_t)t*D);
    __half2 va=v2[lane*2], vb=v2[lane*2+1];
    a0=a0*c+pe*__low2float(va); a1=a1*c+pe*__high2float(va);
    a2=a2*c+pe*__low2float(vb); a3=a3*c+pe*__high2float(vb); m=mn;
  }
  int idx=h*S+s; float* op=Op+(size_t)idx*D+lane*4;
  op[0]=a0; op[1]=a1; op[2]=a2; op[3]=a3; if(lane==0){mp[idx]=m; lp[idx]=l;}
}
// phase 2: one block per head; merge S partials -> O[D], lse.
__global__ void attn_combine(const float* Op,const float* mp,const float* lp,
                             float* O,float* lse,int H,int D,int S){
  int h=blockIdx.x, d=threadIdx.x; float m=-1e30f,l=0.f,acc=0.f;
  for(int s=0;s<S;++s){ int idx=h*S+s; float ms=mp[idx],ls=lp[idx],os=Op[(size_t)idx*D+d];
    float mn=fmaxf(m,ms),c=__expf(m-mn),cs=__expf(ms-mn);
    l=l*c+ls*cs; acc=acc*c+cs*os; m=mn; }
  O[h*D+d]=acc/l; if(d==0) lse[h]=m+logf(l);
}
__global__ void merge2(const float* O0,const float* l0,const float* O1,const float* l1,float* O,int H,int D){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=H*D)return; int h=i/D;
  float a=l0[h],b=l1[h],mx=fmaxf(a,b),w0=__expf(a-mx),w1=__expf(b-mx); O[i]=(O0[i]*w0+O1[i]*w1)/(w0+w1);
}

int main(int argc,char**argv){
  int L=32,H=32,D=128,Ttot=16384,trials=40,S=16;
  if(argc>1)L=atoi(argv[1]); if(argc>2)Ttot=atoi(argv[2]); if(argc>3)trials=atoi(argv[3]); if(argc>4)S=atoi(argv[4]);
  int Tl=Ttot/2,Tp=Ttot-Tl; float scale=1.f/sqrtf((float)D);
  int nd;CK(cudaGetDeviceCount(&nd)); if(nd<2){printf("need 2 GPUs\n");return 1;}
  CK(cudaSetDevice(0));cudaDeviceEnablePeerAccess(1,0); CK(cudaSetDevice(1));cudaDeviceEnablePeerAccess(0,0);
  size_t od=(size_t)H*D, part=(size_t)H*S*D, kvF=(size_t)H*Ttot*D, kv0=(size_t)H*Tl*D, kv1=(size_t)H*Tp*D;
  std::vector<__half*> Kf(L),Vf(L),K0(L),V0(L),K1(L),V1(L),qd(L);
  std::vector<float*> Oout(L),Osg(L),lsg(L),O1h(L),l1h(L);
  // scratch (reused across layers, per device)
  float *spO,*spM,*spL, *spO_p,*spM_p,*spL_p, *O0,*l0sg, *O1,*l1; __half* qp;
  CK(cudaSetDevice(0));
  for(int i=0;i<L;++i){CK(cudaMalloc(&Kf[i],kvF*2));CK(cudaMalloc(&Vf[i],kvF*2));CK(cudaMalloc(&K0[i],kv0*2));CK(cudaMalloc(&V0[i],kv0*2));
    CK(cudaMalloc(&qd[i],od*2));CK(cudaMalloc(&Oout[i],od*4));CK(cudaMalloc(&Osg[i],od*4));CK(cudaMalloc(&lsg[i],H*4));CK(cudaMalloc(&O1h[i],od*4));CK(cudaMalloc(&l1h[i],H*4));
    CK(cudaMemset(Kf[i],0,kvF*2));CK(cudaMemset(Vf[i],0,kvF*2));CK(cudaMemset(K0[i],0,kv0*2));CK(cudaMemset(V0[i],0,kv0*2));CK(cudaMemset(qd[i],0,od*2));}
  CK(cudaMalloc(&spO,part*4));CK(cudaMalloc(&spM,H*S*4));CK(cudaMalloc(&spL,H*S*4));
  CK(cudaMalloc(&O0,od*4));CK(cudaMalloc(&l0sg,H*4));CK(cudaMalloc(&qp,od*2));
  CK(cudaSetDevice(1));
  for(int i=0;i<L;++i){CK(cudaMalloc(&K1[i],kv1*2));CK(cudaMalloc(&V1[i],kv1*2));CK(cudaMemset(K1[i],0,kv1*2));CK(cudaMemset(V1[i],0,kv1*2));}
  CK(cudaMalloc(&spO_p,part*4));CK(cudaMalloc(&spM_p,H*S*4));CK(cudaMalloc(&spL_p,H*S*4));CK(cudaMalloc(&O1,od*4));CK(cudaMalloc(&l1,H*4));
  CK(cudaSetDevice(0));CK(cudaDeviceSynchronize());CK(cudaSetDevice(1));CK(cudaDeviceSynchronize());
  cudaStream_t s0,s1; CK(cudaSetDevice(0));CK(cudaStreamCreate(&s0)); CK(cudaSetDevice(1));CK(cudaStreamCreate(&s1));
  cudaEvent_t eh,ep; CK(cudaSetDevice(0));CK(cudaEventCreate(&eh)); CK(cudaSetDevice(1));CK(cudaEventCreate(&ep));
  int mB=(H*D+127)/128;
  // launch split-K attn on a given device/stream into O/lse
  auto attn=[&](const __half*q_,const __half*K_,const __half*V_,float*O_,float*lse_,int T_,float*pO,float*pM,float*pL,cudaStream_t st){
    attn_split<<<H*S,32,0,st>>>(q_,K_,V_,pO,pM,pL,H,T_,D,S,scale);  // warp/block
    attn_combine<<<H,D,0,st>>>(pO,pM,pL,O_,lse_,H,D,S);
  };
  auto single_layer=[&](int i){ CK(cudaSetDevice(0)); attn(qd[i],Kf[i],Vf[i],Osg[i],lsg[i],Ttot,spO,spM,spL,s0); };
  auto peer_layer=[&](int i){
    CK(cudaSetDevice(0)); CK(cudaMemcpyAsync(qp,qd[i],od*2,cudaMemcpyDefault,s0)); CK(cudaEventRecord(eh,s0));
    CK(cudaSetDevice(1)); CK(cudaStreamWaitEvent(s1,eh,0)); attn(qp,K1[i],V1[i],O1,l1,Tp,spO_p,spM_p,spL_p,s1);
    CK(cudaMemcpyAsync(O1h[i],O1,od*4,cudaMemcpyDefault,s1)); CK(cudaMemcpyAsync(l1h[i],l1,H*4,cudaMemcpyDefault,s1)); CK(cudaEventRecord(ep,s1));
    CK(cudaSetDevice(0)); attn(qd[i],K0[i],V0[i],O0,l0sg,Tl,spO,spM,spL,s0);
    CK(cudaStreamWaitEvent(s0,ep,0)); merge2<<<mB,128,0,s0>>>(O0,l0sg,O1h[i],l1h[i],Oout[i],H,D);
  };
  cudaEvent_t t0,t1; CK(cudaSetDevice(0));CK(cudaEventCreate(&t0));CK(cudaEventCreate(&t1));
  auto bench=[&](auto fn){ for(int w=0;w<8;++w)for(int i=0;i<L;++i)fn(i); CK(cudaSetDevice(0));CK(cudaDeviceSynchronize());CK(cudaSetDevice(1));CK(cudaDeviceSynchronize());
    std::vector<float> ts; for(int r=0;r<trials;++r){CK(cudaSetDevice(0));CK(cudaDeviceSynchronize());CK(cudaSetDevice(1));CK(cudaDeviceSynchronize());
      CK(cudaSetDevice(0));CK(cudaEventRecord(t0,s0)); for(int i=0;i<L;++i)fn(i); CK(cudaEventRecord(t1,s0));
      CK(cudaSetDevice(1));CK(cudaDeviceSynchronize());CK(cudaSetDevice(0));CK(cudaDeviceSynchronize()); float ms;CK(cudaEventElapsedTime(&ms,t0,t1));ts.push_back(ms);} std::sort(ts.begin(),ts.end()); return ts[ts.size()/2]; };
  float sg=bench(single_layer), pe=bench(peer_layer);
  CK(cudaSetDevice(0)); cudaGraph_t g; cudaGraphExec_t ge; bool ok=true;
  if(cudaStreamBeginCapture(s0,cudaStreamCaptureModeGlobal)!=cudaSuccess) ok=false;
  if(ok){for(int i=0;i<L;++i)peer_layer(i); if(cudaStreamEndCapture(s0,&g)!=cudaSuccess)ok=false;}
  if(ok&&cudaGraphInstantiate(&ge,g,0)!=cudaSuccess)ok=false;
  float pg=-1; if(ok){for(int w=0;w<8;++w)CK(cudaGraphLaunch(ge,s0)); CK(cudaSetDevice(0));CK(cudaDeviceSynchronize());
    std::vector<float> ts;for(int r=0;r<trials;++r){CK(cudaDeviceSynchronize());CK(cudaEventRecord(t0,s0));CK(cudaGraphLaunch(ge,s0));CK(cudaEventRecord(t1,s0));CK(cudaDeviceSynchronize());float ms;CK(cudaEventElapsedTime(&ms,t0,t1));ts.push_back(ms);}std::sort(ts.begin(),ts.end());pg=ts[ts.size()/2];}
  printf("Ttot=%-7d S=%-2d | single %8.3f (%.1f us/L) | peer-eager %8.3f (%.2fx) | peer-graph %8.3f (%.2fx)%s\n",
         Ttot,S, sg, sg/L*1000, pe, sg/pe, pg, sg/pg, (sg/pg>=1.0?"  <= single OK":""));
  return 0;
}

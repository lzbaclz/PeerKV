"""e30 -- solve the round-trip with PER-DEVICE CUDA graphs.

e28 showed a SINGLE cross-device graph is uncapturable; e29 showed explicit streams
barely help (1.03x) -- so the cost is per-layer host DISPATCH (many ops) + two
inherent cross-device hops, not missing overlap. The fix that actually applies:
capture each device's per-layer work as its OWN single-device CUDA graph (these DO
capture), so per layer we issue ~2 graph replays + 2 tiny cross-device copies
instead of ~11 eager kernel dispatches.

Per layer i (KV resident, unchanging -> one graph per layer, bound to that layer's
K/V):
  g_peer[i]  (cuda:1): flash over (q1_static, K1[i], V1[i]) -> static (O1,l1)
  g_home[i]  (cuda:0): flash over (q0_static, K0[i], V0[i]) + merge with copied
                       partial -> static out
step: q1_static.copy_(q)  (x-dev) ; g_peer[i].replay ; copy (O1,l1)->cuda:0 (x-dev)
      ; g_home[i].replay
We compare single_gpu, pp eager (shipped), and pp_pdgraph; report exactness.

    python experiments/e30_pdgraph.py --layers 32 --context 16384 --heads 32
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "pdgraph.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import flash_partial, merge_partial, peer_parallel_attention

    if (torch.cuda.device_count() if torch.cuda.is_available() else 0) < 2:
        OUT.write_text(json.dumps({"_is_measured": False, "note": "need 2 GPUs"})); return
    L, C, H, D = args.layers, args.context, args.heads, args.head_dim
    Lc = C // 2; Pc = C - Lc; scale = 1.0 / (D ** 0.5)

    def sync(): torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    def med(xs): xs=sorted(xs); return statistics.median(xs), xs[(3*len(xs))//4]-xs[len(xs)//4]

    qs=[torch.randn(1,H,1,D,dtype=torch.float16,device="cuda:0") for _ in range(L)]
    K0=[torch.randn(1,H,Lc,D,dtype=torch.float16,device="cuda:0") for _ in range(L)]
    V0=[torch.randn(1,H,Lc,D,dtype=torch.float16,device="cuda:0") for _ in range(L)]
    K1=[torch.randn(1,H,Pc,D,dtype=torch.float16,device="cuda:1") for _ in range(L)]
    V1=[torch.randn(1,H,Pc,D,dtype=torch.float16,device="cuda:1") for _ in range(L)]
    Kf=[torch.cat([K0[i],K1[i].to('cuda:0')],dim=2).contiguous() for i in range(L)]
    Vf=[torch.cat([V0[i],V1[i].to('cuda:0')],dim=2).contiguous() for i in range(L)]
    sync()

    # static I/O buffers
    q0_s=torch.zeros(1,H,1,D,dtype=torch.float16,device="cuda:0")
    q1_s=torch.zeros(1,H,1,D,dtype=torch.float16,device="cuda:1")
    O1h_s=torch.zeros(1,H,1,D,dtype=torch.float16,device="cuda:0")
    l1h_s=torch.zeros(1,H,1,dtype=torch.float32,device="cuda:0")

    g_peer=[]; g_home=[]; peer_out=[]; home_out=[]
    capture_ok=True; err=None
    try:
        # ---- capture per-layer peer graphs on cuda:1 ----
        torch.cuda.set_device(1)
        for i in range(L):
            s=torch.cuda.Stream(device=1); s.wait_stream(torch.cuda.current_stream(1))
            with torch.cuda.stream(s):
                for _ in range(3): flash_partial(q1_s,K1[i],V1[i],scale)
            torch.cuda.current_stream(1).wait_stream(s); torch.cuda.synchronize(1)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                O1c,l1c=flash_partial(q1_s,K1[i],V1[i],scale)
            g_peer.append(g); peer_out.append((O1c,l1c))
        # ---- capture per-layer home graphs on cuda:0 (flash + merge) ----
        torch.cuda.set_device(0)
        for i in range(L):
            s=torch.cuda.Stream(device=0); s.wait_stream(torch.cuda.current_stream(0))
            with torch.cuda.stream(s):
                for _ in range(3):
                    O0,l0=flash_partial(q0_s,K0[i],V0[i],scale); merge_partial(O0,l0,O1h_s,l1h_s)
            torch.cuda.current_stream(0).wait_stream(s); torch.cuda.synchronize(0)
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                O0,l0=flash_partial(q0_s,K0[i],V0[i],scale)
                outc,_=merge_partial(O0,l0,O1h_s,l1h_s)
            g_home.append(g); home_out.append(outc)
        sync()
    except Exception as ex:
        capture_ok=False; err=str(ex)[:160]
        print(f"  per-device graph capture FAILED: {err}")

    def pp_pdgraph(i):
        q1_s.copy_(qs[i], non_blocking=True)        # cuda:0 -> cuda:1 (x-dev)
        g_peer[i].replay()                          # cuda:1 flash
        O1c,l1c=peer_out[i]
        O1h_s.copy_(O1c, non_blocking=True)         # cuda:1 -> cuda:0 (x-dev)
        l1h_s.copy_(l1c, non_blocking=True)
        q0_s.copy_(qs[i], non_blocking=True)
        g_home[i].replay()                          # cuda:0 flash + merge
        return home_out[i]

    def loop(fn):
        for i in range(L): fn(i)
    def bench(fn, warmup=5):
        for _ in range(warmup): loop(fn)
        sync(); s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);ts=[]
        for _ in range(args.trials):
            sync(); s.record(); loop(fn); e.record(); sync(); ts.append(s.elapsed_time(e))
        return med(ts)

    res={"_experiment":"e30_pdgraph","_is_measured":True,"layers":L,"context":C,
         "heads":H,"head_dim":D,"trials":args.trials,"device":torch.cuda.get_device_name(0),
         "capture_ok":capture_ok}
    sg_ms,_=bench(lambda i: F.scaled_dot_product_attention(qs[i],Kf[i],Vf[i],scale=scale))
    base_ms,_=bench(lambda i: peer_parallel_attention(qs[i],[(K0[i],V0[i]),(K1[i],V1[i])],scale=scale))
    res["single_gpu_ms"]=sg_ms; res["pp_baseline_ms"]=base_ms
    print(f"  single_gpu     {sg_ms:.3f} ms ({sg_ms/L*1000:.1f} us/layer)")
    print(f"  pp_baseline    {base_ms:.3f} ms ({base_ms/L*1000:.1f} us/layer)  {sg_ms/base_ms:.2f}x vs single")
    if capture_ok:
        pd_ms,_=bench(pp_pdgraph)
        res.update({"pp_pdgraph_ms":pd_ms,"pdgraph_vs_single":sg_ms/pd_ms,
                    "pdgraph_vs_baseline":base_ms/pd_ms})
        # exactness
        ref=F.scaled_dot_product_attention(qs[0],Kf[0],Vf[0],scale=scale)
        out=pp_pdgraph(0); sync()
        cos=float(F.cosine_similarity(ref.reshape(-1).float(),out.reshape(-1).float(),dim=0))
        res["pdgraph_exactness_cos"]=cos
        print(f"  pp_pdgraph     {pd_ms:.3f} ms ({pd_ms/L*1000:.1f} us/layer)  "
              f"{sg_ms/pd_ms:.2f}x vs single, {base_ms/pd_ms:.2f}x vs baseline  cos={cos:.6f}")
    else:
        res["capture_error"]=err
    res["_generated_at"]=datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(res,indent=2)); print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

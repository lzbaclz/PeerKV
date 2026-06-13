"""e25 -- FAIR head-to-head decode step (the reviewer-hardened headline artifact).

Every arm attends over the SAME total KV; arms differ only in WHERE the KV lives
and WHAT crosses the link. All baselines get the SAME favourable treatment
(coalesced + one-step-ahead double-buffered overlap), so the comparison cannot be
accused of a strided/synchronous straw-man (the e20 host arm used a non-contiguous
strided copy -> 3.6 GB/s; here host is contiguous + overlapped).

Arms:
  * single_gpu        : full KV on cuda:0, fused flash SDPA (reference; OOMs at scale)
  * peer_parallel     : KV sharded local/peer; compute partial where KV lives, move
                        only the ~KB (O,lse) over NVLink (ours)
  * copyback_overlap  : cold KV resident on peer (cuda:1), streamed back to cuda:0 in
                        C-block chunks with double-buffered prefetch, online-merged
                        (bounded peak) -- the FAIR copy-back
  * host_overlap      : same, cold KV on host pinned (contiguous) -- the FAIR host

Per arm we report median latency + IQR (trials x seeds), aggregate and PER-GPU
throughput, and ISOLATED cuda:0 peak (each arm allocated alone). Configs: balanced
(1024/1024) and overflow (128/1920). One JSON per config; every cell _is_measured.

    python experiments/e25_fair_decode.py --trials 20 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def _write(name: str, res: dict) -> None:
    res["_experiment"] = "e25_fair_decode"
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    res["_is_measured"] = True
    RES.mkdir(parents=True, exist_ok=True)
    (RES / name).write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {RES / name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=2048)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunk-blocks", type=int, default=128)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import flash_partial, merge_partial

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        _write("e25_balanced.json", {"_is_measured": False,
                                     "note": f"need >=2 GPUs (have {ngpu})"})
        return

    H, T, D = args.heads, args.block_tokens, args.head_dim
    N, C = args.n_blocks, max(1, args.chunk_blocks)
    scale = 1.0 / (D ** 0.5)
    Ctok = C * T
    blk_kv_mb = (T * D * 2 * H) * 2 / 1024**2          # one block K+V, MiB
    dev = "cuda:0"

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med_iqr(xs):
        xs = sorted(xs)
        q1 = xs[len(xs)//4]; q3 = xs[(3*len(xs))//4]
        return statistics.median(xs), (q3 - q1)

    # event-timed wrapper, trials x seeds
    def timed(step, fresh, warmup=5):
        ms = []
        for _ in range(args.seeds):
            fresh()
            for _ in range(warmup):
                step()
            sync()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            for _ in range(args.trials):
                sync(); s.record(); step(); e.record(); sync()
                ms.append(s.elapsed_time(e))
        return med_iqr(ms)

    def run_config(L):
        P = N - L
        tag = f"L{L}_P{P}"
        print(f"\n=== config {tag}: {L} local / {P} peer "
              f"({N*blk_kv_mb/1024:.2f} GB total KV) ===")
        out = {"n_blocks": N, "local_blocks": L, "peer_blocks": P,
               "block_tokens": T, "heads": H, "head_dim": D, "chunk_blocks": C,
               "trials": args.trials, "seeds": args.seeds,
               "total_kv_gb": N * blk_kv_mb / 1024,
               "device": torch.cuda.get_device_name(0),
               "peer_partial_bytes_per_layer": H*D*2 + H*4}
        q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")

        # ---------- peer_parallel ----------
        K0 = torch.randn(1, H, L*T, D, dtype=torch.float16, device="cuda:0")
        V0 = torch.randn(1, H, L*T, D, dtype=torch.float16, device="cuda:0")
        K1 = torch.randn(1, H, P*T, D, dtype=torch.float16, device="cuda:1")
        V1 = torch.randn(1, H, P*T, D, dtype=torch.float16, device="cuda:1")
        sync()

        def pp_step():
            q1 = q.to("cuda:1", non_blocking=True)
            O1, l1 = flash_partial(q1, K1, V1, scale)
            O0, l0 = flash_partial(q, K0, V0, scale)
            O1c = O1.to("cuda:0", non_blocking=True); l1c = l1.to("cuda:0", non_blocking=True)
            o, _ = merge_partial(O0, l0, O1c, l1c)
            return o

        # exactness vs dense, self-contained (the online-softmax merge is
        # config-independent, but we measure cos for THIS config so the artifact
        # stands on its own rather than borrowing another file's number).
        with torch.no_grad():
            Kf = torch.cat([K0, K1.to("cuda:0")], dim=2).float()
            Vf = torch.cat([V0, V1.to("cuda:0")], dim=2).float()
            ref = F.scaled_dot_product_attention(q.float(), Kf, Vf, scale=scale)
            ppo = pp_step().float()
            out["exactness_cosine"] = float(F.cosine_similarity(
                ref.reshape(-1), ppo.reshape(-1), dim=0))
            out["exactness_max_abs_err"] = float((ref - ppo).abs().max())
            print(f"  exactness       cos={out['exactness_cosine']:.6f}  "
                  f"max_abs={out['exactness_max_abs_err']:.2e}")
            del Kf, Vf, ref, ppo; torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(0)
        m, iqr = timed(pp_step, lambda: None)
        out["peer_parallel"] = {"ms": m, "iqr_ms": iqr,
                                "peak_mb_cuda0": torch.cuda.max_memory_allocated(0)/1024**2}
        print(f"  peer_parallel   {m:7.3f} ms (IQR {iqr:.3f})  peak {out['peer_parallel']['peak_mb_cuda0']:.0f} MB")

        # ---------- single_gpu (full KV on cuda:0) ----------
        try:
            Kall = torch.cat([K0, K1.to("cuda:0")], dim=2)
            Vall = torch.cat([V0, V1.to("cuda:0")], dim=2)
            def sg_step():
                return F.scaled_dot_product_attention(q, Kall, Vall, scale=scale)
            torch.cuda.reset_peak_memory_stats(0)
            m, iqr = timed(sg_step, lambda: None)
            out["single_gpu"] = {"ms": m, "iqr_ms": iqr,
                                 "peak_mb_cuda0": torch.cuda.max_memory_allocated(0)/1024**2}
            print(f"  single_gpu      {m:7.3f} ms (IQR {iqr:.3f})")
            del Kall, Vall
        except RuntimeError as e:
            out["single_gpu"] = {"status": f"OOM: {str(e)[:60]}"}
            print(f"  single_gpu      OOM")
        torch.cuda.empty_cache()

        # ---------- fair overlapped copy-back (peer) and host ----------
        def make_chunks(spill_dev):
            ks, vs = [], []
            for s in range(0, P*T, Ctok):
                n = min(Ctok, P*T - s)
                if spill_dev == "cuda:1":
                    ks.append(K1[:, :, s:s+n, :].contiguous())
                    vs.append(V1[:, :, s:s+n, :].contiguous())
                else:  # host pinned, contiguous
                    ks.append(torch.randn(1, H, n, D, dtype=torch.float16).pin_memory())
                    vs.append(torch.randn(1, H, n, D, dtype=torch.float16).pin_memory())
            return ks, vs

        copy_stream = torch.cuda.Stream(device="cuda:0")

        def overlap_step(ks, vs):
            O, l = flash_partial(q, K0, V0, scale)            # local
            n = len(ks)
            buf = [None]*n; ev = [torch.cuda.Event() for _ in range(n)]
            with torch.cuda.stream(copy_stream):
                buf[0] = (ks[0].to("cuda:0", non_blocking=True), vs[0].to("cuda:0", non_blocking=True))
                ev[0].record(copy_stream)
            for i in range(n):
                if i+1 < n:
                    with torch.cuda.stream(copy_stream):
                        buf[i+1] = (ks[i+1].to("cuda:0", non_blocking=True),
                                    vs[i+1].to("cuda:0", non_blocking=True))
                        ev[i+1].record(copy_stream)
                torch.cuda.current_stream().wait_event(ev[i])
                Kc, Vc = buf[i]
                Oc, lc = flash_partial(q, Kc, Vc, scale)
                O, l = merge_partial(O, l, Oc, lc)
                buf[i] = None
            return O

        for arm, spill in (("copyback_overlap", "cuda:1"), ("host_overlap", "host")):
            ks, vs = make_chunks(spill)
            torch.cuda.reset_peak_memory_stats(0)
            m, iqr = timed(lambda: overlap_step(ks, vs), lambda: None)
            out[arm] = {"ms": m, "iqr_ms": iqr,
                        "peak_mb_cuda0": torch.cuda.max_memory_allocated(0)/1024**2}
            print(f"  {arm:16s}{m:7.3f} ms (IQR {iqr:.3f})  peak {out[arm]['peak_mb_cuda0']:.0f} MB")
            del ks, vs; torch.cuda.empty_cache()

        # ---------- derived: speedups + throughput ----------
        pp = out["peer_parallel"]["ms"]
        def sp(arm):
            a = out.get(arm, {})
            return (a["ms"]/pp) if a.get("ms") else None
        out["speedup_vs_copyback_overlap"] = sp("copyback_overlap")
        out["speedup_vs_host_overlap"] = sp("host_overlap")
        out["speedup_vs_single_gpu"] = sp("single_gpu")
        # throughput honesty: peer-parallel uses 2 GPUs' compute
        agg = 1000.0/pp
        out["peer_parallel_tok_s_aggregate"] = agg
        out["peer_parallel_tok_s_per_gpu"] = agg/2
        if out.get("single_gpu", {}).get("ms"):
            sg = 1000.0/out["single_gpu"]["ms"]
            out["single_gpu_tok_s"] = sg
            out["per_gpu_throughput_ratio_vs_single"] = (agg/2)/sg   # <1 => latency trade
            out["aggregate_if_two_independent_single"] = 2*sg
        del K0, V0, K1, V1; torch.cuda.empty_cache()
        _write(f"e25_{tag}.json", out)
        return out

    bal = run_config(args.n_blocks // 2)            # balanced
    ovf = run_config(min(128, args.n_blocks // 16)) # overflow
    print("\n=== SUMMARY ===")
    for c in (bal, ovf):
        print(f"  L={c['local_blocks']:>4}/P={c['peer_blocks']:<4}  pp={c['peer_parallel']['ms']:.2f}ms"
              f"  vs_copyback={c['speedup_vs_copyback_overlap']:.1f}x"
              f"  vs_host={c['speedup_vs_host_overlap']:.1f}x"
              f"  vs_single={c['speedup_vs_single_gpu']}"
              f"  per_gpu_tput_ratio={c.get('per_gpu_throughput_ratio_vs_single')}")


if __name__ == "__main__":
    main()

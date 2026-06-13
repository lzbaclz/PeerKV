"""e46 -- end-to-end KV handoff with overlap-safe bandwidth + clean victim TPOT."""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

from overlap_safe_bw import drain_stream, measure_bw_events, start_sustained_load

OUT = Path(__file__).resolve().parent / "results" / "handoff_e2e.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=65536)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from umallm.transfer.direction import Endpoint, choose_initiator

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    S = args.ctx
    dt = torch.float16
    torch.cuda.set_device(1)
    Wq = torch.randn(D, H * HD, dtype=dt, device="cuda:1") * 0.02
    Wk = torch.randn(D, HKV * HD, dtype=dt, device="cuda:1") * 0.02
    Wv = torch.randn(D, HKV * HD, dtype=dt, device="cuda:1") * 0.02
    Wo = torch.randn(H * HD, D, dtype=dt, device="cuda:1") * 0.02
    Wg = torch.randn(D, DFF, dtype=dt, device="cuda:1") * 0.02
    Wu = torch.randn(D, DFF, dtype=dt, device="cuda:1") * 0.02
    Wd = torch.randn(DFF, D, dtype=dt, device="cuda:1") * 0.02

    n = args.handoff_mb * 1024 * 1024 // 2
    nbytes = n * 2
    kv_src = torch.randn(n, dtype=dt, device="cuda:1")
    kv_dst = torch.empty(n, dtype=dt, device="cuda:0")
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)
    holder_ls = torch.cuda.Stream(device=1)

    def copy_pull():
        with torch.cuda.stream(s0):
            kv_dst.copy_(kv_src, non_blocking=True)

    def copy_push():
        with torch.cuda.stream(s1):
            kv_dst.copy_(kv_src, non_blocking=True)

    def handoff_bw(copy_fn, copy_stream, decode_step):
        start_sustained_load(decode_step, holder_ls)
        gbps, _ = measure_bw_events(
            copy_fn, copy_stream, nbytes, args.trials,
            holder_stream=holder_ls, holder_op=decode_step, assert_holder_busy=True,
        )
        drain_stream(holder_ls)
        return gbps

    results = {
        "_experiment": "e46_handoff_e2e",
        "_is_measured": True,
        "_timing_method": "overlap_safe_cuda_events",
        "device": torch.cuda.get_device_name(0),
        "geometry": "Llama-3-8B GQA (random weights)",
        "ctx": S,
        "handoff_mb": args.handoff_mb,
        "seeds": args.seeds,
        "by_batch": [],
    }

    for B in args.batches:
        Kc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
        Vc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
        x = torch.randn(B, 1, D, dtype=dt, device="cuda:1") * 0.02

        def decode_step():
            q = (x @ Wq).view(B, 1, H, HD).transpose(1, 2)
            k = (x @ Wk).view(B, 1, HKV, HD).transpose(1, 2)
            v = (x @ Wv).view(B, 1, HKV, HD).transpose(1, 2)
            K = torch.cat([Kc, k], dim=2)
            V = torch.cat([Vc, v], dim=2)
            o = F.scaled_dot_product_attention(q, K, V, enable_gqa=True)
            a = (o.transpose(1, 2).reshape(B, 1, H * HD) @ Wo)
            return a + (F.silu(x @ Wg) * (x @ Wu)) @ Wd

        torch.cuda.set_device(1)
        for _ in range(5):
            decode_step()
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        for _ in range(30):
            decode_step()
        torch.cuda.synchronize(1)
        tpot_alone = (time.perf_counter() - t0) / 30 * 1e3
        torch.cuda.set_device(0)

        pull_vals, push_vals = [], []
        victim_pull_vals, victim_push_vals = [], []
        for _ in range(args.seeds):
            pull_vals.append(handoff_bw(copy_pull, s0, decode_step))
            push_vals.append(handoff_bw(copy_push, s1, decode_step))

            # victim TPOT: decode + copy truly concurrent (unchanged, clean)
            torch.cuda.set_device(1)
            for _ in range(3):
                decode_step()
                with torch.cuda.stream(s0):
                    kv_dst.copy_(kv_src, non_blocking=True)
                with torch.cuda.stream(s1):
                    kv_dst.copy_(kv_src, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                decode_step()
                with torch.cuda.stream(s0):
                    kv_dst.copy_(kv_src, non_blocking=True)
            torch.cuda.synchronize()
            victim_pull = (time.perf_counter() - t0) / 30 * 1e3

            t0 = time.perf_counter()
            for _ in range(30):
                decode_step()
                with torch.cuda.stream(s1):
                    kv_dst.copy_(kv_src, non_blocking=True)
            torch.cuda.synchronize()
            victim_push = (time.perf_counter() - t0) / 30 * 1e3
            torch.cuda.set_device(0)

            victim_pull_vals.append((victim_pull / tpot_alone - 1) * 100)
            victim_push_vals.append((victim_push / tpot_alone - 1) * 100)

        bw_pull = statistics.mean(pull_vals)
        bw_push = statistics.mean(push_vals)
        mem_bound = True
        init = choose_initiator(Endpoint(1, mem_bound), Endpoint(0, False))
        bw_sel = bw_push if init == 1 else bw_pull
        bw_oracle = max(bw_pull, bw_push)

        row = {
            "batch": B,
            "tpot_alone_ms": round(tpot_alone, 2),
            "bw_pull": round(bw_pull, 1),
            "bw_push": round(bw_push, 1),
            "bw_selector": round(bw_sel, 1),
            "bw_oracle": round(bw_oracle, 1),
            "selector_dir": "push" if init == 1 else "pull",
            "push_adv_pp": round((bw_push / bw_pull - 1) * 100, 1),
            "victim_tpot_pull_pct": round(statistics.mean(victim_pull_vals), 1),
            "victim_tpot_push_pct": round(statistics.mean(victim_push_vals), 1),
            "victim_tpot_inflation_pct": round(statistics.mean(victim_push_vals), 1),
        }
        results["by_batch"].append(row)
        print(f"  B={B:3d} tpot_alone={tpot_alone:6.1f}ms  pull={bw_pull:5.0f} push={bw_push:5.0f} "
              f"push_adv=+{row['push_adv_pp']:.0f}pp  "
              f"victim pull/push={row['victim_tpot_pull_pct']:.1f}%/{row['victim_tpot_push_pct']:.1f}%")
        del Kc, Vc, x
        torch.cuda.empty_cache()

    results["selector_matches_oracle"] = all(
        abs(r["bw_selector"] - r["bw_oracle"]) / r["bw_oracle"] < 0.03 for r in results["by_batch"]
    )
    results["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(results, indent=2))
    print(f"  selector matches oracle: {results['selector_matches_oracle']}")
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

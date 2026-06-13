"""e12 -- Phase 2 enablement: peak / TPOT vs context, stock vs tiered.

For each context length we run the SAME model/prompt under stock mlx-lm and
under UMA-LLM's tiered attention, recording MLX peak memory, decode TPOT, and
the tiered cache's at-rest KV footprint. The headline RQ2 evidence: the tiered
peak curve grows slower than stock, pushing the OOM boundary to the right on a
fixed-memory (32 GB) machine.

    python experiments/e12_context_sweep.py \
        --model mlx-community/Llama-3.2-1B-Instruct-4bit \
        --contexts 2048,4096,8192,16384 --n-gen 8
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "context_sweep.json"


def _prompt(tok, n):
    base = ("Unified memory accelerators share one physical pool between CPU "
            "and GPU; we study KV-cache residency when demotion is in-place "
            "quantization rather than a bus transfer. ")
    s = base
    while len(tok.encode(s)) < n:
        s += base
    ids = tok.encode(s)[:n]
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--contexts", default="2048,4096,8192,16384")
    ap.add_argument("--n-gen", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-sink-blocks", type=int, default=1)
    ap.add_argument("--n-window-blocks", type=int, default=2)
    ap.add_argument("--cold-bits", type=int, default=4)
    ap.add_argument("--include-mlxkv4", action="store_true",
                    help="also measure mlx-lm all-or-nothing QuantizedKVCache "
                         "(kv-bits=4); off by default (its quantized_kv_start=0 "
                         "peak is a known config artifact -- see e13 for a fair "
                         "quality comparison instead)")
    ap.add_argument("--out", default="context_sweep.json")
    args = ap.parse_args()
    global OUT
    OUT = Path(__file__).resolve().parent / "results" / args.out

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from umallm.mlx_tiered_attention import MLXTieredCache, enable_tiered_attention

    print(f"[load] {args.model}")
    model, tokenizer = load(args.model)
    n_layers = len(make_prompt_cache(model))
    enable_tiered_attention(model)
    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]

    def decode(make_cache, kv_bits=None):
        mx.reset_peak_memory()
        last = None
        t0 = time.perf_counter()
        cache = make_cache()
        kw = {"kv_bits": kv_bits, "kv_group_size": 64,
              "quantized_kv_start": 0} if kv_bits else {}
        for r in stream_generate(model, tokenizer, prompt, max_tokens=args.n_gen,
                                 prompt_cache=cache, **kw):
            last = r
        gtps = getattr(last, "generation_tps", None) or float("nan")
        peak = float(getattr(last, "peak_memory", 0.0)) or mx.get_peak_memory() / 1024**3
        return {"tpot_ms": (1000.0 / gtps) if gtps == gtps and gtps else float("nan"),
                "peak_gb": peak, "wall_s": time.perf_counter() - t0}, cache

    def make_tiered():
        return [MLXTieredCache(block_size=args.block_size,
                               n_sink_blocks=args.n_sink_blocks,
                               n_window_blocks=args.n_window_blocks,
                               cold_bits=args.cold_bits, group_size=64)
                for _ in range(n_layers)]

    rows = []
    for ctx in contexts:
        prompt = _prompt(tokenizer, ctx)
        b, _ = decode(lambda: make_prompt_cache(model))                 # stock fp16
        t, tcache = decode(make_tiered)                                 # ours: tiered
        fp = tcache[0].footprint()
        resident = sum(c.nbytes for c in tcache) / 1e6
        full = sum(c.footprint()["fp16_bytes"] for c in tcache) / 1e6
        row = {"ctx": ctx,
               "stock_peak_gb": b["peak_gb"], "tiered_peak_gb": t["peak_gb"],
               "peak_ratio_vs_stock": t["peak_gb"] / b["peak_gb"] if b["peak_gb"] else None,
               "stock_tpot_ms": b["tpot_ms"], "tiered_tpot_ms": t["tpot_ms"],
               "tiered_resident_kv_mb": resident, "full_kv_mb": full,
               "kv_compression_x": (full / resident) if resident else 1.0,
               "cold_tokens": fp["cold_tokens"]}
        if args.include_mlxkv4:
            qk, _ = decode(lambda: make_prompt_cache(model), kv_bits=4)
            row["mlx_kv4_peak_gb"] = qk["peak_gb"]
            row["mlx_kv4_tpot_ms"] = qk["tpot_ms"]
        rows.append(row)
        print(f"ctx={ctx:>6}  peak stock={b['peak_gb']:.2f} tiered={t['peak_gb']:.2f} "
              f"({row['peak_ratio_vs_stock']:.2f}x)  "
              f"tpot {b['tpot_ms']:.1f}->{t['tpot_ms']:.1f} ms  "
              f"KV {full:.0f}->{resident:.0f}MB ({row['kv_compression_x']:.2f}x)")

    res = {"_experiment": "e12_context_sweep", "_is_measured": True,
           "model": args.model, "n_layers": n_layers,
           "block_size": args.block_size, "n_window_blocks": args.n_window_blocks,
           "cold_bits": args.cold_bits, "n_gen": args.n_gen,
           "host": "Apple M2 Pro 32GB",
           "rows": rows, "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

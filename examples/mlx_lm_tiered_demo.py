"""Phase 1 end-to-end: tiered attention inside real mlx-lm generation.

Loads a model, installs the block-wise tiered attention (hot fp16 + cold
low-bit, online-softmax merge), and compares against stock mlx-lm on the same
model/prompt:
  (A) logit fidelity  (teacher-forced forward: tiered vs stock)
  (B) at-rest KV footprint / compression_x
  (C) decode TPOT + MLX peak memory  -- now expected ~1.x stock (not 28x), and
      peak bounded because the cold tier is never materialized as fp16.

Usage:
    python examples/mlx_lm_tiered_demo.py \
        --model mlx-community/Llama-3.2-1B-Instruct-4bit \
        --prompt-tokens 4096 --n-gen 32 --block-size 128 --n-window-blocks 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _peak_gb():
    import mlx.core as mx
    return float(mx.get_peak_memory() / 1024 ** 3)


def _reset_peak():
    import mlx.core as mx
    mx.reset_peak_memory()


def _long_prompt(tok, n):
    base = ("Unified memory accelerators expose a single physical pool shared "
            "by CPU and GPU; this work studies how to manage the KV cache when "
            "there is no PCIe transfer and the costs are coherence and in-place "
            "dequantization. ")
    s = base
    while len(tok.encode(s)) < n:
        s += base
    return s


def _win_logits_np(model, toks, cache, window):
    import mlx.core as mx
    out = model(toks, cache=cache)
    mx.eval(out)
    w = min(window, out.shape[1])
    a = np.array(out[0, -w:, :].astype(mx.float32))
    del out
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--n-gen", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-sink-blocks", type=int, default=1)
    ap.add_argument("--n-window-blocks", type=int, default=4)
    ap.add_argument("--cold-bits", type=int, default=4)
    ap.add_argument("--fidelity-window", type=int, default=512)
    args = ap.parse_args()

    import mlx.core as mx  # noqa: F401
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from umallm.kv_runtime import logit_fidelity
    from umallm.mlx_tiered_attention import MLXTieredCache, enable_tiered_attention

    print(f"[load] {args.model}")
    model, tokenizer = load(args.model)
    n_layers = len(make_prompt_cache(model))
    enable_tiered_attention(model)   # safe: only MLXTieredCache routes to tiered
    prompt = _long_prompt(tokenizer, args.prompt_tokens)

    def make_tiered():
        return [MLXTieredCache(block_size=args.block_size,
                               n_sink_blocks=args.n_sink_blocks,
                               n_window_blocks=args.n_window_blocks,
                               cold_bits=args.cold_bits, group_size=64)
                for _ in range(n_layers)]

    res = {"_is_measured": True,
           "scope": "real model + real mlx-lm engine; block-wise tiered "
                    "attention (hot fp16 + cold low-bit, online-softmax merge).",
           "model": args.model, "n_layers": n_layers,
           "block_size": args.block_size, "n_sink_blocks": args.n_sink_blocks,
           "n_window_blocks": args.n_window_blocks, "cold_bits": args.cold_bits}

    # (A)+(B) fidelity + footprint via a teacher-forced forward
    print("[A/B] teacher-forced forward: stock vs tiered ...")
    ids = tokenizer.encode(prompt)
    toks = mx.array([ids])
    fb = _win_logits_np(model, toks, make_prompt_cache(model), args.fidelity_window)
    tiered = make_tiered()
    fu = _win_logits_np(model, toks, tiered, args.fidelity_window)
    res["fidelity"] = logit_fidelity(fb, fu)
    per_layer = tiered[0].footprint()
    resident = sum(c.nbytes for c in tiered)
    full = sum(c.footprint()["fp16_bytes"] for c in tiered)
    res["footprint"] = {"n_prompt_tokens": len(ids),
                        "resident_kv_mb": resident / 1e6,
                        "full_kv_mb": full / 1e6,
                        "compression_x": (full / resident) if resident else 1.0,
                        "per_layer": per_layer}
    print(f"  prompt={len(ids)} tok  rel_l2={res['fidelity']['rel_l2']:.4f} "
          f"cos={res['fidelity']['mean_cosine']:.4f} "
          f"argmax={res['fidelity']['argmax_agreement']:.3f}")
    print(f"  at-rest KV {resident/1e6:.1f} MB vs {full/1e6:.1f} MB "
          f"(compression_x={res['footprint']['compression_x']:.2f}); "
          f"cold_tok={per_layer['cold_tokens']}")

    # (C) decode TPOT + peak
    def decode(make_cache):
        _reset_peak()
        last = None
        t0 = time.perf_counter()
        for r in stream_generate(model, tokenizer, prompt, max_tokens=args.n_gen,
                                 prompt_cache=make_cache()):
            last = r
        gtps = getattr(last, "generation_tps", None) or float("nan")
        return {"tpot_ms": (1000.0 / gtps) if gtps == gtps and gtps else float("nan"),
                "generation_tps": gtps,
                "peak_gb": float(getattr(last, "peak_memory", 0.0)) or _peak_gb(),
                "wall_s": time.perf_counter() - t0}

    print("[C] decode TPOT + peak ...")
    res["baseline_decode"] = decode(lambda: make_prompt_cache(model))
    res["tiered_decode"] = decode(make_tiered)
    b, t = res["baseline_decode"], res["tiered_decode"]
    print(f"  baseline {b['tpot_ms']:.1f} ms/tok @ {b['peak_gb']:.2f} GB")
    print(f"  tiered   {t['tpot_ms']:.1f} ms/tok @ {t['peak_gb']:.2f} GB "
          f"({t['tpot_ms']/b['tpot_ms']:.2f}x decode)")

    out = (Path(__file__).resolve().parent.parent / "experiments" / "results"
           / "mac_tiered_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()

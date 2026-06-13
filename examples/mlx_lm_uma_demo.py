"""Run UMA-LLM end-to-end inside mlx-lm on Apple Silicon and MEASURE it honestly.

Compares stock mlx-lm ``KVCache`` against ``umallm.UMALayerCache`` on the SAME
model, reporting three things:

  (A) Logit fidelity -- one teacher-forced forward over a long prompt: how
      much the 4-bit cold tier perturbs the *real model's* logits
      (rel_l2, mean cosine, argmax agreement). This is the quality cost.

  (B) At-rest KV footprint -- resident cache bytes and compression_x under the
      residency policy. This is the metric the compression tier is designed to
      move (fit more KV in the fixed unified-memory pool).

  (C) Decode TPOT + MLX peak memory -- reported HONESTLY. The dense
      reconstruct-on-fetch path rebuilds the full tensor every step, so it does
      NOT lower the live attention peak and it makes decode slower (per-step
      numpy dequant). This number documents that cost. Lowering the live peak
      requires block-wise / quantized-SDPA attention that never materializes
      the full tensor -- the paper flags that as future work.

Results -> experiments/results/mac_mlx_lm_demo.json.

Usage (on your Mac):
    pip install -e . && pip install mlx-lm
    python examples/mlx_lm_uma_demo.py \
        --model mlx-community/Llama-3.2-1B-Instruct-4bit \
        --prompt-tokens 1536 --n-gen 16 --block-size 128 --n-active 4
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


def _long_prompt(tokenizer, target_tokens):
    base = (
        "Unified memory architectures such as Apple Silicon and NVIDIA "
        "Grace-Hopper expose a single physical memory pool shared by the CPU "
        "and GPU. This work studies how the key-value cache of a large language "
        "model should be managed when there is no PCIe transfer cost and the "
        "dominant costs are cold cache-line touches and in-place dequantization. "
    )
    text = base
    while len(tokenizer.encode(text)) < target_tokens:
        text += base
    return text


def _window_logits_np(model, tokens_mx, cache, window):
    """Run one forward and return last-`window` logits as fp32 numpy."""
    import mlx.core as mx
    out = model(tokens_mx, cache=cache)          # (1, N, vocab)
    mx.eval(out)
    w = min(window, out.shape[1])
    arr = np.array(out[0, -w:, :].astype(mx.float32))
    del out
    return arr


def measure_fidelity(model, tokenizer, prompt, make_uma, window):
    """Teacher-forced single forward: stock vs UMA logits over last `window`."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from umallm.kv_runtime import logit_fidelity

    ids = tokenizer.encode(prompt)
    toks = mx.array([ids])                        # (1, N)

    fb = _window_logits_np(model, toks, make_prompt_cache(model), window)
    uma_cache = make_uma()
    fu = _window_logits_np(model, toks, uma_cache, window)

    fid = logit_fidelity(fb, fu)
    per_layer = uma_cache[0].footprint()
    resident = sum(c.nbytes for c in uma_cache)
    full = sum(c.footprint()["fp16_bytes"] for c in uma_cache)
    summary = {
        "n_prompt_tokens": len(ids),
        "resident_bytes_all_layers": int(resident),
        "fp16_bytes_all_layers": int(full),
        "compression_x_all_layers": (full / resident) if resident else 1.0,
    }
    return fid, per_layer, summary


def measure_decode(model, tokenizer, prompt, make_cache, n_gen):
    """Decode TPOT + peak via stream_generate's per-token stats."""
    from mlx_lm import stream_generate
    _reset_peak()
    last = None
    t0 = time.perf_counter()
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=n_gen,
                                prompt_cache=make_cache()):
        last = resp
    wall = time.perf_counter() - t0
    gtps = getattr(last, "generation_tps", None) or float("nan")
    return {
        "generation_tps": gtps,
        "tpot_ms": (1000.0 / gtps) if gtps == gtps and gtps else float("nan"),
        "prompt_tps": getattr(last, "prompt_tps", float("nan")),
        "peak_gb": (float(last.peak_memory) if getattr(last, "peak_memory", None)
                    else _peak_gb()),
        "wall_s": wall,
        "n_gen": getattr(last, "generation_tokens", n_gen),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--prompt-tokens", type=int, default=1536)
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--cold-bits", type=int, default=4)
    ap.add_argument("--n-active", type=int, default=4)
    ap.add_argument("--fidelity-window", type=int, default=256)
    args = ap.parse_args()

    import mlx.core as mx  # noqa: F401  (import side effects / availability)
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from umallm.mlx_backend import UMALayerCache
    from umallm.policy import UMAPolicy

    print(f"[load] {args.model}")
    model, tokenizer = load(args.model)
    n_layers = len(make_prompt_cache(model))
    prompt = _long_prompt(tokenizer, args.prompt_tokens)

    def make_uma():
        pol = UMAPolicy(n_active=args.n_active, n_sink=1, n_window=2)
        return [UMALayerCache(block_size=args.block_size, cold_bits=args.cold_bits,
                              policy=pol) for _ in range(n_layers)]

    result = {
        "_is_measured": True,
        "scope": ("real model + real mlx-lm engine on Apple Silicon; "
                  "fidelity & at-rest footprint are the demonstrated result. "
                  "TPOT/peak reported honestly: dense reconstruct-on-fetch does "
                  "NOT lower live peak and is slower -- block-wise attention is "
                  "future work."),
        "model": args.model, "n_layers": n_layers,
        "block_size": args.block_size, "cold_bits": args.cold_bits,
        "n_active": args.n_active,
    }

    print("[A/B] teacher-forced forward: stock vs UMA (fidelity + footprint) ...")
    fid, per_layer, fp_all = measure_fidelity(model, tokenizer, prompt, make_uma,
                                              args.fidelity_window)
    result["fidelity"] = fid
    result["footprint_per_layer"] = per_layer
    result["footprint_all_layers"] = fp_all
    print(f"  prompt={fp_all['n_prompt_tokens']} tok  "
          f"rel_l2={fid['rel_l2']:.4f}  cos={fid['mean_cosine']:.4f}  "
          f"argmax_agree={fid['argmax_agreement']:.3f}")
    print(f"  at-rest KV: compression_x={fp_all['compression_x_all_layers']:.2f}  "
          f"({fp_all['resident_bytes_all_layers']/1e6:.1f} MB resident vs "
          f"{fp_all['fp16_bytes_all_layers']/1e6:.1f} MB full)  "
          f"cold={per_layer['n_cold']}/{per_layer['n_blocks']} blocks")

    print("[C] decode TPOT + peak (honest) ...")
    result["baseline_decode"] = measure_decode(model, tokenizer, prompt,
                                               lambda: make_prompt_cache(model),
                                               args.n_gen)
    result["uma_decode"] = measure_decode(model, tokenizer, prompt, make_uma,
                                          args.n_gen)
    b, u = result["baseline_decode"], result["uma_decode"]
    print(f"  baseline {b['tpot_ms']:.1f} ms/tok @ {b['peak_gb']:.2f} GB peak")
    print(f"  uma      {u['tpot_ms']:.1f} ms/tok @ {u['peak_gb']:.2f} GB peak "
          f"(slower by design; documents the reconstruct cost)")

    out = (Path(__file__).resolve().parent.parent / "experiments" / "results"
           / "mac_mlx_lm_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()

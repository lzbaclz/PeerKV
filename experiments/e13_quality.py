"""e13 -- quality of tiered vs all-or-nothing KV quantization (RQ3/RQ4).

Teacher-forced forward over a real passage; compare, against the stock fp16
cache, the perplexity and top-1 agreement of:
  * tiered 4-bit  (ours: recent tokens fp16, cold 4-bit)
  * tiered 2-bit  (ours)
  * mlx-lm all-4bit QuantizedKVCache  (all-or-nothing, the must-beat baseline)
The point: tiering keeps the *recent* (high-attention) tokens in fp16, so it
should preserve quality better than quantizing the whole cache at the same bit
width. Perplexity over the last `--ppl-window` positions (the cold-affected
tail); top-1 agreement vs the fp16 cache's argmax.

    python experiments/e13_quality.py --model mlx-community/Llama-3.2-1B-Instruct-4bit
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "quality.json"

# A self-contained natural-prose passage (varied paragraphs); repeated to reach
# the target length. Relative ppl across cache configs is the comparison and is
# fair under repetition.
_PARAS = [
    "The history of computing is a history of memory. Each generation of "
    "machines was defined less by how fast it could compute than by how much "
    "state it could hold close to the processor, and at what cost that state "
    "could be moved. From mercury delay lines to magnetic cores to dynamic "
    "random-access memory, the hierarchy grew taller and the gaps between its "
    "levels grew wider.",
    "Long-context language models revived this old tension in a new form. The "
    "attention mechanism must consult, in principle, every previous token, and "
    "so the key-value cache it maintains grows without bound as a conversation "
    "lengthens. On a machine with a fixed pool of memory, the question is not "
    "how to compute attention quickly but how to keep its growing footprint "
    "from spilling past the edge of what the hardware can hold.",
    "Unified memory changes the shape of the answer. When the processor and "
    "the accelerator share a single physical pool, there is no bus to cross "
    "and no copy to schedule; a block of cache is the same bytes at the same "
    "address whoever reads it. The cost that remains is the cost of touching "
    "cold data and of paying, in precision, for the room to keep more of it "
    "resident at once.",
    "A tier, in this setting, is no longer a place but a state. A block may be "
    "warm in the caches of whichever unit last touched it; it may be resident "
    "but compressed to a few bits each; or it may have been pushed out to the "
    "slow backing store the operating system keeps in reserve. The art is to "
    "decide, continuously and cheaply, which blocks deserve which state.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-window-blocks", type=int, default=2)
    ap.add_argument("--ppl-window", type=int, default=512)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, QuantizedKVCache
    from umallm.mlx_tiered_attention import MLXTieredCache, enable_tiered_attention

    print(f"[load] {args.model}")
    model, tok = load(args.model)
    n_layers = len(make_prompt_cache(model))
    enable_tiered_attention(model)

    text = ""
    while len(tok.encode(text)) < args.ctx:
        text += " ".join(_PARAS) + " "
    ids = tok.encode(text)[:args.ctx]
    toks = mx.array([ids])
    W = min(args.ppl_window, len(ids) - 1)

    def logits_np(cache):
        out = model(toks, cache=cache)
        mx.eval(out)
        return np.array(out[0].astype(mx.float32))  # (N, V)

    def metrics(lg, ref_argmax=None):
        # ppl over the last W predicted positions; targets are ids[i+1]
        sl = lg[-(W + 1):-1, :]                       # (W, V) predicting ids[-W:]
        tgt = np.array(ids[-W:])
        sl = sl - sl.max(axis=-1, keepdims=True)
        logp = sl - np.log(np.exp(sl).sum(axis=-1, keepdims=True))
        nll = -logp[np.arange(W), tgt]
        ppl = float(np.exp(nll.mean()))
        am = lg[-(W + 1):-1, :].argmax(axis=-1)
        agree = None if ref_argmax is None else float((am == ref_argmax).mean())
        return ppl, am, agree

    def tiered(bits):
        return [MLXTieredCache(block_size=args.block_size, n_sink_blocks=1,
                               n_window_blocks=args.n_window_blocks,
                               cold_bits=bits, group_size=64) for _ in range(n_layers)]

    print(f"[forward] ctx={len(ids)} ppl-window={W}")
    lg_ref = logits_np(make_prompt_cache(model))
    ppl_ref, am_ref, _ = metrics(lg_ref)

    configs = {
        "tiered_4bit": tiered(4),
        "tiered_2bit": tiered(2),
        "mlx_all_4bit": [QuantizedKVCache(group_size=64, bits=4) for _ in range(n_layers)],
    }
    res = {"_experiment": "e13_quality", "_is_measured": True,
           "model": args.model, "ctx": len(ids), "ppl_window": W,
           "block_size": args.block_size, "n_window_blocks": args.n_window_blocks,
           "host": "Apple M2 Pro 32GB",
           "fp16_reference": {"ppl": ppl_ref, "top1_agreement": 1.0}}
    for name, cache in configs.items():
        lg = logits_np(cache)
        ppl, _, agree = metrics(lg, am_ref)
        kv_mb = None
        if name.startswith("tiered"):
            kv_mb = sum(c.nbytes for c in cache) / 1e6
        res[name] = {"ppl": ppl, "ppl_ratio_vs_fp16": ppl / ppl_ref,
                     "top1_agreement_vs_fp16": agree,
                     "resident_kv_mb": kv_mb}
        print(f"  {name:<14} ppl={ppl:7.3f} (x{ppl/ppl_ref:.3f} fp16)  "
              f"top1={agree:.4f}" + (f"  KV={kv_mb:.0f}MB" if kv_mb else ""))
    print(f"  {'fp16_reference':<14} ppl={ppl_ref:7.3f}")

    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

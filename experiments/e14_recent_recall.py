"""e14 -- does keeping the recent window in fp16 help *recent* exact recall?

The one place tiered KV could beat all-or-nothing quantization: it keeps the
recent (sliding-window) tokens in fp16, while mlx-lm's QuantizedKVCache
quantizes *everything*. If a task needs to read a token in the recent window
exactly, 4-bit there might hurt where our fp16 window does not.

Probe: embed a random CODE in the context, then teacher-force the same CODE at
the very end and measure, per CODE token, the model's top-1 hit rate and mean
log-prob -- a finer signal than greedy correctness. Two needle positions:
  * near  -- inside the recent window (we keep fp16; mlx quantizes)
  * far   -- in a cold block (both quantize; control)
Three caches: fp16 (ref), tiered (window fp16 + cold 4-bit), mlx all-4bit.

If tiered>=fp16>mlx-4bit at the *near* position, tiering has a real,
quality-side win. If all three tie, it does not -- report honestly.

    python experiments/e14_recent_recall.py --n 24
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "recent_recall.json"

_ALPH = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_WORDS = ("river mountain copper silver garden window harbor candle forest "
          "anchor velvet meadow lantern pepper marble thunder orchard pebble "
          "willow saddle ribbon glacier compass bramble").split()
_FILLER = ("The document continues with general background material that is "
           "not relevant to the question, describing routine procedures and "
           "unremarkable observations at some length so that the context is "
           "padded to the desired size. ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-window-blocks", type=int, default=2)
    ap.add_argument("--code-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, QuantizedKVCache
    from umallm.mlx_tiered_attention import MLXTieredCache, enable_tiered_attention

    print(f"[load] {args.model}")
    model, tok = load(args.model)
    n_layers = len(make_prompt_cache(model))
    enable_tiered_attention(model)
    rng = np.random.default_rng(args.seed)
    window_tok = args.n_window_blocks * args.block_size

    fill_ids = tok.encode(_FILLER)

    def pad_to(ids, n):
        while len(ids) < n:
            ids = ids + fill_ids
        return ids[:n]

    def build(code, position):
        """Return (ids, code_token_positions) for a near/far needle."""
        needle = tok.encode(f"\nThe secret access code is {code}.\n")
        qpref = tok.encode("\nThe secret access code is ")
        code_ids = tok.encode(code)
        tail = qpref + code_ids                       # teacher-forced target = code_ids
        if position == "near":
            # needle ends ~ (len(tail)+gap) tokens from the end -> within window
            gap = max(0, window_tok - len(tail) - len(needle) - 24)
            head = pad_to([], args.ctx - len(tail) - gap - len(needle))
            ids = head + needle + pad_to([], gap) + tail
        else:  # far: needle near the front (a cold block)
            head = tok.encode("Document.\n")
            mid = pad_to([], args.ctx - len(tail) - len(needle) - len(head))
            ids = head + needle + mid + tail
        ids = ids[:args.ctx]
        cpos = list(range(len(ids) - len(code_ids), len(ids)))
        return ids, cpos, code_ids

    def score(ids, cpos, code_ids, make_cache):
        toks = mx.array([ids])
        out = model(toks, cache=make_cache())
        mx.eval(out)
        lg = out[0]                                   # (N, V)
        hits, lps = 0, []
        for j, p in enumerate(cpos):
            row = lg[p - 1].astype(mx.float32)        # predicts token at p
            tgt = int(code_ids[j])
            row = row - mx.logsumexp(row)
            lps.append(float(row[tgt]))
            hits += int(int(mx.argmax(row)) == tgt)
        return hits / len(cpos), float(np.mean(lps))

    def tiered():
        return [MLXTieredCache(block_size=args.block_size, n_sink_blocks=1,
                               n_window_blocks=args.n_window_blocks,
                               cold_bits=4, group_size=64) for _ in range(n_layers)]
    caches = {
        "fp16": lambda: make_prompt_cache(model),
        "tiered_4bit": tiered,
        "mlx_all_4bit": lambda: [QuantizedKVCache(group_size=64, bits=4)
                                 for _ in range(n_layers)],
    }

    agg = {pos: {c: {"hit": [], "lp": []} for c in caches} for pos in ("near", "far")}
    for i in range(args.n):
        code = " ".join(rng.choice(_WORDS, size=3))   # copyable word-phrase
        for pos in ("near", "far"):
            ids, cpos, code_ids = build(code, pos)
            for cname, mk in caches.items():
                h, lp = score(ids, cpos, code_ids, mk)
                agg[pos][cname]["hit"].append(h)
                agg[pos][cname]["lp"].append(lp)
        if (i + 1) % 6 == 0:
            print(f"  ...{i+1}/{args.n}")

    res = {"_experiment": "e14_recent_recall", "_is_measured": True,
           "model": args.model, "n": args.n, "ctx": args.ctx,
           "code_len_chars": args.code_len, "window_tok": window_tok,
           "host": "Apple M2 Pro 32GB", "results": {}}
    print(f"\n{'pos/cache':<22}{'top1_hit':>10}{'mean_logprob':>14}")
    for pos in ("near", "far"):
        res["results"][pos] = {}
        for c in caches:
            hit = float(np.mean(agg[pos][c]["hit"]))
            lp = float(np.mean(agg[pos][c]["lp"]))
            res["results"][pos][c] = {"top1_hit": hit, "mean_logprob": lp}
            print(f"{pos+'/'+c:<22}{hit:>10.3f}{lp:>14.3f}")
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()

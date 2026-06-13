"""e9 -- TRUE end-to-end decode through UMA-LLM (runs on CPU, real numbers).

Unlike e7 (which structures the MLX/70B headline matrix that needs a Mac),
this driver actually exercises the full UMA-LLM decode path in the sandbox
using the NumPy reference model: token -> Q/K/V -> append to the tiered
cache -> residency policy compresses cold blocks -> attention reads the
*reconstructed* (dequantized) K/V -> logits -> next token. It is the
end-to-end integration test the project was missing.

It reports three things that are all real here:
  * **memory**: KV footprint with tiering vs. the fp16 no-tiering baseline,
    and the residency-tier histogram over the run.
  * **fidelity**: teacher-forced logit agreement between the full-precision
    cache and the 4-bit / 2-bit tiered caches -- i.e. how much the residency
    policy perturbs the output end-to-end (the cost of the cold tier).
  * **work**: how many block compressions the policy triggered.

What this does NOT measure: wall-clock TPOT or real-model task quality --
those need MLX weights on Apple Silicon (see e7 and the mlx_backend drop-in).
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

import numpy as np  # noqa: E402

from . import add_repo_to_path, save_result  # noqa: E402

add_repo_to_path()

from umallm.backends.numpy_ref import RefTransformer  # noqa: E402
from umallm.kv_runtime import (  # noqa: E402
    TieredKVCache, greedy_decode, tiered_score_sequence, logit_fidelity,
)
from umallm.policy import UMAPolicy  # noqa: E402


def _make_cache(model, block_size, n_active, cold_bits, compress):
    pol = UMAPolicy(n_active=n_active, n_sink=1, n_window=1,
                    gpu_frac=0.7, headroom_frac=0.1)
    return TieredKVCache(
        n_layers=model.n_layers, n_heads=model.n_heads, head_dim=model.head_dim,
        block_size=block_size, policy=pol, cold_bits=cold_bits, compress=compress,
    )


def run(prompt_len: int = 48, n_gen: int = 16, block_size: int = 8,
        n_active: int = 3, seed: int = 0) -> dict:
    model = RefTransformer(n_layers=2, n_heads=4, head_dim=32, vocab=128, seed=seed)
    rng = np.random.default_rng(seed)
    prompt = rng.integers(0, model.vocab, size=prompt_len).tolist()

    # 1) reference trajectory from the full-precision cache (greedy).
    ref_cache = _make_cache(model, block_size, n_active, 4, compress=False)
    gen = greedy_decode(model, prompt, n_gen, ref_cache)
    seq = prompt + gen  # the fixed token sequence we score everywhere

    # 2) teacher-force the same sequence through three caches.
    full = tiered_score_sequence(model, seq, _make_cache(model, block_size, n_active, 4, False))
    t4_cache = _make_cache(model, block_size, n_active, 4, True)
    t4 = tiered_score_sequence(model, seq, t4_cache)
    t2_cache = _make_cache(model, block_size, n_active, 2, True)
    t2 = tiered_score_sequence(model, seq, t2_cache)

    fid4 = logit_fidelity(full, t4)
    fid2 = logit_fidelity(full, t2)
    fp4 = t4_cache.footprint()
    fp2 = t2_cache.footprint()

    payload = {
        "_is_measured": True,  # real end-to-end pipeline execution on CPU
        "scope": "pipeline/integration E2E on CPU reference model; "
                 "real-model TPOT+quality need MLX on Apple Silicon (see e7)",
        "config": {"prompt_len": prompt_len, "n_gen": n_gen,
                   "block_size": block_size, "n_active": n_active,
                   "seq_len": len(seq), "tokens_decoded": len(gen)},
        "tiered_4bit": {
            "fidelity": fid4,
            "compression_x": fp4["compression_x"],
            "tier_block_counts": fp4["tier_block_counts"],
            "block_compressions": fp4["n_decode_compress"],
            "kv_bytes_actual": fp4["actual_bytes"],
            "kv_bytes_fp16": fp4["fp16_bytes"],
        },
        "tiered_2bit": {
            "fidelity": fid2,
            "compression_x": fp2["compression_x"],
            "kv_bytes_actual": fp2["actual_bytes"],
            "kv_bytes_fp16": fp2["fp16_bytes"],
        },
        "ran_end_to_end": True,
    }
    return payload


def main() -> dict:
    p = run()
    c = p["config"]
    print("=== e9 end-to-end UMA-LLM decode (CPU reference, real numbers) ===")
    print(f"  decoded {c['tokens_decoded']} tokens; scored seq_len={c['seq_len']} "
          f"(block_size={c['block_size']}, n_active={c['n_active']})")
    for name, key in [("4-bit cold tier", "tiered_4bit"), ("2-bit cold tier", "tiered_2bit")]:
        d = p[key]
        f = d["fidelity"]
        print(f"  [{name}] KV {d['kv_bytes_fp16']}B fp16 -> {d['kv_bytes_actual']}B "
              f"({d['compression_x']:.2f}x);  cosine={f['mean_cosine']:.4f} "
              f"rel_l2={f['rel_l2']:.4f} argmax_agree={f['argmax_agreement']:.2f}")
    print(f"  tier histogram (4bit) = {p['tiered_4bit']['tier_block_counts']} "
          f"(compressions={p['tiered_4bit']['block_compressions']})")
    path = save_result("e9_e2e_uma_decode", p)
    print(f"  -> {path}")
    return p


if __name__ == "__main__":
    main()

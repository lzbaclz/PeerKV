"""e7 -- End-to-end enablement headline (Claim C4).

Structures the Llama-3 enablement matrix that is the abstract's headline:
on a 96 GB M2 Max, does UMA-LLM let Llama-3-70B (Q4) run within the
unified-memory budget where stock mlx-lm OOMs / swaps? We build the full
(model x context x config) matrix and fill each cell with
:class:`umallm.baselines.mlx_lm_baseline.MLXLMBaseline`.

Off-Mac, ``MLXLMBaseline.run`` returns ``_is_measured=False`` placeholders
(NaN TPOT / peak memory), so every row is clearly marked as awaiting the
real run. The matrix shape, model list, context ladder, and the two configs
(``vanilla`` baseline vs ``uma_llm``) are fixed here so the Mac run only has
to swap the placeholder numbers in.

Configs:
  * ``vanilla``  -- stock mlx-lm KV cache (the B1 baseline; expected to
    OOM/swap on 70B at long context on a 96 GB Mac).
  * ``uma_llm``  -- UMA-LLM tiering enabled (the cell we expect to keep
    running with a bounded P99 TPOT). Until the integration lands on a Mac
    this shares the same sandbox placeholder, flagged ``_is_measured=False``.
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

from . import add_repo_to_path, save_result  # noqa: E402

add_repo_to_path()

from umallm.baselines.mlx_lm_baseline import MLXLMBaseline  # noqa: E402

MODELS = [
    {"id": "mlx-community/Meta-Llama-3-8B-Instruct-4bit", "tag": "Llama-3-8B-Q4"},
    {"id": "mlx-community/Meta-Llama-3-70B-Instruct-4bit", "tag": "Llama-3-70B-Q4"},
]
CONTEXTS = [8192, 32768, 65536]
CONFIGS = ["vanilla", "uma_llm"]
TARGET_P99_TPOT_MS = 27.0  # ICCD plan headline target for 70B


def run(contexts: list[int] | None = None, n_gen: int = 64) -> dict:
    contexts = contexts or CONTEXTS
    baseline = MLXLMBaseline()
    measured = baseline.available()

    rows = []
    for model in MODELS:
        for n_ctx in contexts:
            for config in CONFIGS:
                # Both configs currently route through the same baseline
                # wrapper off-hardware (placeholder). On a Mac, ``uma_llm``
                # would run with tiering enabled; the row schema is identical
                # so the result drops straight into the table.
                res = baseline.run(model["id"], n_ctx=n_ctx, n_gen=n_gen)
                d = res.to_dict()
                rows.append(
                    {
                        "model": model["tag"],
                        "config": config,
                        "n_ctx": n_ctx,
                        "p50_tpot_ms": d["p50_tpot_ms"],
                        "p99_tpot_ms": d["p99_tpot_ms"],
                        "peak_mem_gb": d["peak_mem_gb"],
                        "oom_or_swap": d["oom_or_swap"],
                        "_is_measured": d["_is_measured"],
                    }
                )

    payload = {
        "_is_measured": measured,  # False off-Mac -> placeholder matrix
        "target_p99_tpot_ms": TARGET_P99_TPOT_MS,
        "host": "M2_Max_96GB (target)",
        "models": [m["tag"] for m in MODELS],
        "contexts": contexts,
        "configs": CONFIGS,
        "matrix": rows,
        "n_rows": len(rows),
        "n_placeholder_rows": sum(1 for r in rows if not r["_is_measured"]),
        "note": (
            "Off-Mac placeholders (NaN). Real M2-Max run fills p99_tpot_ms / "
            "peak_mem_gb and flips oom_or_swap for the vanilla 70B long-context "
            "cells."
        ),
    }
    return payload


def main() -> dict:
    payload = run()
    print("=== e7 e2e enablement (headline) ===")
    print(f"  _is_measured = {payload['_is_measured']} "
          f"(placeholders={payload['n_placeholder_rows']}/{payload['n_rows']})")
    print("  model            config   n_ctx   P99-TPOT(ms)  peak(GB)  OOM/swap  measured")
    for r in payload["matrix"]:
        p99 = "nan" if r["p99_tpot_ms"] != r["p99_tpot_ms"] else f"{r['p99_tpot_ms']:.2f}"
        mem = "nan" if r["peak_mem_gb"] != r["peak_mem_gb"] else f"{r['peak_mem_gb']:.1f}"
        print(
            f"  {r['model']:15s} {r['config']:7s} {r['n_ctx']:6d}   {p99:>11s}   "
            f"{mem:>7s}   {str(r['oom_or_swap']):8s} {str(r['_is_measured'])}"
        )
    print(f"  target P99 TPOT (70B) = {payload['target_p99_tpot_ms']} ms (fill on Mac)")
    path = save_result("e7_e2e_enablement", payload)
    print(f"  -> {path}")
    return payload


if __name__ == "__main__":
    main()

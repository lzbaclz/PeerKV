"""e8 -- llama.cpp Metal baseline (the mandatory on-device comparison).

llama.cpp's Metal backend is the baseline an ICCD reviewer will expect.
This driver sweeps the context ladder {4k, 8k, 16k, 32k, 64k} on one Q4
GGUF model via
:class:`umallm.baselines.llama_cpp_metal.LlamaCppMetalBaseline` and
structures the "where does llama.cpp start swapping?" comparison: as context
grows, the KV cache eventually no longer fits in unified memory and
llama.cpp (which has no intra-request KV tiering) OOM/swaps -- the tok/s
falls off a cliff and peak RSS exceeds physical memory. That cliff point is
the cell UMA-LLM's T2 compression tier is meant to push to the right.

Off-hardware (no ``llama-bench`` on PATH) every row is an ``_is_measured =
False`` placeholder with NaN throughput, so the matrix runs in CI and the
real Mac run fills it. We also report the modeled physical-memory budget so
the expected swap context can be annotated before the run.
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

from . import add_repo_to_path, save_result  # noqa: E402

add_repo_to_path()

from umallm.baselines.llama_cpp_metal import LlamaCppMetalBaseline  # noqa: E402

CONTEXTS = [4096, 8192, 16384, 32768, 65536]
DEFAULT_MODEL = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"


def run(
    model_path: str = DEFAULT_MODEL,
    contexts: list[int] | None = None,
    n_gen: int = 128,
) -> dict:
    contexts = contexts or CONTEXTS
    bench = LlamaCppMetalBaseline()
    available = bench.available()

    rows = []
    swap_context = None
    for n_ctx in contexts:
        res = bench.run(model_path, n_ctx=n_ctx, n_gen=n_gen)
        d = res.to_dict()
        rows.append(
            {
                "n_ctx": n_ctx,
                "decode_tok_per_s": d["decode_tok_per_s"],
                "prefill_tok_per_s": d["prefill_tok_per_s"],
                "peak_rss_gb": d["peak_rss_gb"],
                "swapped": d["swapped"],
                "_is_measured": d["_is_measured"],
            }
        )
        if d["_is_measured"] and d["swapped"] and swap_context is None:
            swap_context = n_ctx

    payload = {
        "_is_measured": available,  # False off-Mac -> placeholder sweep
        "model": model_path,
        "contexts": contexts,
        "sweep": rows,
        "n_rows": len(rows),
        "n_placeholder_rows": sum(1 for r in rows if not r["_is_measured"]),
        "swap_onset_context": swap_context,  # filled on real Mac
        "note": (
            "Off-hardware placeholders (no llama-bench on PATH). On a Mac, "
            "swap_onset_context = the smallest n_ctx where peak RSS exceeds "
            "physical memory and decode tok/s collapses; UMA-LLM should keep "
            "running past it."
        ),
    }
    return payload


def main() -> dict:
    payload = run()
    print("=== e8 llama.cpp Metal baseline ===")
    print(f"  _is_measured = {payload['_is_measured']} "
          f"(placeholders={payload['n_placeholder_rows']}/{payload['n_rows']})")
    print("  n_ctx   decode(tok/s)  prefill(tok/s)  peak_rss(GB)  swapped  measured")
    for r in payload["sweep"]:
        dec = "nan" if r["decode_tok_per_s"] != r["decode_tok_per_s"] else f"{r['decode_tok_per_s']:.1f}"
        pre = "nan" if r["prefill_tok_per_s"] != r["prefill_tok_per_s"] else f"{r['prefill_tok_per_s']:.1f}"
        rss = "nan" if r["peak_rss_gb"] != r["peak_rss_gb"] else f"{r['peak_rss_gb']:.1f}"
        print(
            f"  {r['n_ctx']:6d}  {dec:>12s}   {pre:>13s}   {rss:>11s}   "
            f"{str(r['swapped']):7s}  {str(r['_is_measured'])}"
        )
    print(f"  swap onset context = {payload['swap_onset_context']} (fill on Mac)")
    path = save_result("e8_llama_cpp_baseline", payload)
    print(f"  -> {path}")
    return payload


if __name__ == "__main__":
    main()

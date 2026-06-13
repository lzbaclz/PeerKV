"""Baselines for UMA-LLM evaluation.

Two reference points the paper must beat or match on Apple Silicon:

* :mod:`umallm.baselines.llama_cpp_metal` -- llama.cpp's Metal backend, the
  *de-facto* on-device baseline (addresses SOTA gap: prior UMA-LLM drafts
  did not compare to it).
* :mod:`umallm.baselines.mlx_lm_baseline` -- stock mlx-lm cache (no tiering),
  the B1 row in the evaluation.

Both wrappers are sandbox-safe: when the underlying binary / framework is
absent they return a clearly-marked placeholder (``_is_measured=False``) so
experiment drivers run end-to-end off-hardware and the real numbers drop in
on a Mac.
"""

from .llama_cpp_metal import LlamaCppMetalBaseline, LlamaCppResult
from .mlx_lm_baseline import MLXLMBaseline, MLXLMResult

__all__ = [
    "LlamaCppMetalBaseline",
    "LlamaCppResult",
    "MLXLMBaseline",
    "MLXLMResult",
]

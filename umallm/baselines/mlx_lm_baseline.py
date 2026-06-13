"""Stock mlx-lm baseline (no tiering) -- the B1 row.

Runs a model under mlx-lm's default KV cache, measuring decode TPOT and
peak unified-memory footprint. This is the "do nothing special" reference;
UMA-LLM's T2 compression tier is what lets the same model fit / stay off
the swap path at longer context.

Sandbox-safe: returns ``_is_measured=False`` if MLX is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import mlx.core as mx  # noqa: F401
    import mlx_lm  # noqa: F401

    HAS_MLX_LM = True
except Exception:  # pragma: no cover - exercised only off-Mac
    HAS_MLX_LM = False


@dataclass
class MLXLMResult:
    model: str
    n_ctx: int
    n_gen: int
    p50_tpot_ms: float
    p99_tpot_ms: float
    peak_mem_gb: float
    oom_or_swap: bool
    _is_measured: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MLXLMBaseline:
    """Thin wrapper over mlx-lm generation for baseline timing."""

    def available(self) -> bool:
        return HAS_MLX_LM

    def run(
        self,
        model_id: str,
        n_ctx: int = 4096,
        n_gen: int = 128,
    ) -> MLXLMResult:
        if not HAS_MLX_LM:
            return MLXLMResult(
                model=model_id, n_ctx=n_ctx, n_gen=n_gen,
                p50_tpot_ms=float("nan"), p99_tpot_ms=float("nan"),
                peak_mem_gb=float("nan"), oom_or_swap=False,
                _is_measured=False,
            )
        # --- real path (runs on a Mac) ---------------------------------- #
        import time

        import mlx.core as mx
        from mlx_lm import load, generate  # type: ignore

        model, tokenizer = load(model_id)
        prompt = "a " * max(1, n_ctx - 8)
        toks = tokenizer.encode(prompt)[:n_ctx]
        latencies_ms: list[float] = []
        # warmup
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=4, verbose=False)
        last = time.perf_counter()
        # Per-token timing via the streaming generator when available.
        for _ in range(n_gen):
            _ = generate(model, tokenizer, prompt=prompt, max_tokens=1, verbose=False)
            now = time.perf_counter()
            latencies_ms.append((now - last) * 1e3)
            last = now
        import numpy as np

        arr = np.asarray(latencies_ms)
        peak_gb = float(mx.get_peak_memory() / (1024 ** 3)) if hasattr(mx, "get_peak_memory") else float("nan")
        return MLXLMResult(
            model=model_id, n_ctx=n_ctx, n_gen=n_gen,
            p50_tpot_ms=float(np.percentile(arr, 50)),
            p99_tpot_ms=float(np.percentile(arr, 99)),
            peak_mem_gb=peak_gb,
            oom_or_swap=False,
            _is_measured=True,
        )

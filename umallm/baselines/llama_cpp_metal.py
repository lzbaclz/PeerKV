"""llama.cpp Metal-backend baseline.

llama.cpp is the de-facto on-device LLM runtime on Apple Silicon, and its
Metal backend is the baseline an ICCD reviewer will expect UMA-LLM to be
compared against. This module shells out to ``llama-bench`` (preferred, it
emits machine-readable throughput) or ``llama-cli`` and parses
prompt/decode throughput and peak resident memory.

Design notes
------------
* We do **not** vendor llama.cpp; the caller points us at a built
  ``llama-bench`` binary and a GGUF model. On a machine without either,
  :meth:`run` returns a placeholder with ``_is_measured=False`` so the
  experiment harness runs off-hardware and the paper's tables stay
  consistent until the Mac run lands.
* llama.cpp has no intra-request KV tiering; at long context it either fits
  in unified memory or it OOM/swaps. That is exactly the baseline UMA-LLM's
  T2 compression tier is meant to beat, so the interesting cell is the
  context length at which llama.cpp (at a given quant) starts swapping.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class LlamaCppResult:
    model: str
    n_ctx: int
    decode_tok_per_s: float
    prefill_tok_per_s: float
    peak_rss_gb: float
    swapped: bool
    _is_measured: bool = True
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "n_ctx": self.n_ctx,
            "decode_tok_per_s": self.decode_tok_per_s,
            "prefill_tok_per_s": self.prefill_tok_per_s,
            "peak_rss_gb": self.peak_rss_gb,
            "swapped": self.swapped,
            "_is_measured": self._is_measured,
        }


@dataclass
class LlamaCppMetalBaseline:
    """Wrapper around a built ``llama-bench`` / ``llama-cli``."""

    llama_bench_bin: str = "llama-bench"
    n_gpu_layers: int = 999  # offload all layers to Metal
    n_threads: int = 8
    extra_args: list[str] = field(default_factory=list)

    def available(self) -> bool:
        return shutil.which(self.llama_bench_bin) is not None

    def run(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gen: int = 128,
        timeout_s: float = 600.0,
    ) -> LlamaCppResult:
        """Benchmark one (model, context) cell.

        Returns a placeholder (``_is_measured=False``) if ``llama-bench`` is
        not on PATH, so callers can run the full experiment matrix in CI.
        """
        if not self.available():
            return self._placeholder(model_path, n_ctx)
        cmd = [
            self.llama_bench_bin,
            "-m", model_path,
            "-ngl", str(self.n_gpu_layers),
            "-t", str(self.n_threads),
            "-p", str(n_ctx),
            "-n", str(n_gen),
            "-o", "json",
            *self.extra_args,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, check=True
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            out = getattr(exc, "output", "") or str(exc)
            return self._placeholder(model_path, n_ctx, raw=out, swapped=True)
        return self._parse(proc.stdout, model_path, n_ctx)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(stdout: str, model_path: str, n_ctx: int) -> LlamaCppResult:
        """Parse ``llama-bench -o json`` output.

        llama-bench emits a JSON list of result objects with keys such as
        ``avg_ts`` (tokens/s) tagged by test type ``pp`` (prompt) / ``tg``
        (text-gen). We fall back to a regex over human-readable output.
        """
        prefill = decode = 0.0
        try:
            rows = json.loads(stdout)
            for r in rows:
                ts = float(r.get("avg_ts", 0.0))
                kind = str(r.get("test", r.get("n_prompt", "")))
                if "pp" in kind or str(r.get("n_prompt", 0)) not in ("0", ""):
                    prefill = max(prefill, ts)
                else:
                    decode = max(decode, ts)
        except json.JSONDecodeError:
            for m in re.finditer(r"(pp|tg)\d+\s*\|\s*([\d.]+)\s*tokens", stdout):
                if m.group(1) == "pp":
                    prefill = float(m.group(2))
                else:
                    decode = float(m.group(2))
        return LlamaCppResult(
            model=model_path.rsplit("/", 1)[-1],
            n_ctx=n_ctx,
            decode_tok_per_s=decode,
            prefill_tok_per_s=prefill,
            peak_rss_gb=_peak_rss_gb(),
            swapped=False,
            _is_measured=True,
            raw=stdout[:2000],
        )

    @staticmethod
    def _placeholder(
        model_path: str, n_ctx: int, raw: str = "", swapped: bool = False
    ) -> LlamaCppResult:
        return LlamaCppResult(
            model=model_path.rsplit("/", 1)[-1],
            n_ctx=n_ctx,
            decode_tok_per_s=float("nan"),
            prefill_tok_per_s=float("nan"),
            peak_rss_gb=float("nan"),
            swapped=swapped,
            _is_measured=False,
            raw=raw,
        )


def _peak_rss_gb() -> float:
    """Best-effort peak resident set size of this process, in GB."""
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        import sys

        scale = 1 if sys.platform == "darwin" else 1024
        return ru * scale / (1024 ** 3)
    except Exception:
        return float("nan")

"""Tests for the sandbox-safe baseline wrappers.

Both wrappers must run off-hardware (no llama-bench, no MLX) and return
``_is_measured=False`` placeholders rather than crashing, and the
llama-bench JSON parser must handle a representative ``-o json`` payload.
"""
import math

from umallm.baselines.llama_cpp_metal import LlamaCppMetalBaseline, LlamaCppResult
from umallm.baselines.mlx_lm_baseline import HAS_MLX_LM, MLXLMBaseline, MLXLMResult


def test_llama_cpp_placeholder_off_hardware():
    b = LlamaCppMetalBaseline(llama_bench_bin="definitely-not-a-real-binary-xyz")
    assert b.available() is False
    res = b.run("some/model.gguf", n_ctx=8192)
    assert isinstance(res, LlamaCppResult)
    assert res._is_measured is False
    assert res.n_ctx == 8192
    assert math.isnan(res.decode_tok_per_s)
    # to_dict round-trips the placeholder flag.
    d = res.to_dict()
    assert d["_is_measured"] is False and d["n_ctx"] == 8192


def test_mlx_lm_placeholder_off_mac():
    b = MLXLMBaseline()
    if not HAS_MLX_LM:
        # Off-Mac / no MLX: must return a placeholder rather than crash.
        assert b.available() is False
        res = b.run("mlx-community/Meta-Llama-3-8B-Instruct-4bit", n_ctx=4096, n_gen=8)
        assert isinstance(res, MLXLMResult)
        assert res._is_measured is False
        assert math.isnan(res.p99_tpot_ms)
        assert res.oom_or_swap is False
        d = res.to_dict()
        assert d["_is_measured"] is False
    else:
        # On a real Mac with MLX present the wrapper reports availability; we
        # do not download an 8B model inside a unit test.
        assert b.available() is True


def test_llama_bench_json_parsing():
    """The static parser handles a representative llama-bench JSON string."""
    sample = (
        '[{"model_filename": "m.gguf", "n_prompt": 512, "n_gen": 0, '
        '"test": "pp512", "avg_ts": 950.5}, '
        '{"model_filename": "m.gguf", "n_prompt": 0, "n_gen": 128, '
        '"test": "tg128", "avg_ts": 42.7}]'
    )
    res = LlamaCppMetalBaseline._parse(sample, "path/to/m.gguf", n_ctx=512)
    assert isinstance(res, LlamaCppResult)
    assert res._is_measured is True
    assert res.model == "m.gguf"
    # prefill (pp) is the larger throughput; decode (tg) the smaller.
    assert res.prefill_tok_per_s == 950.5
    assert res.decode_tok_per_s == 42.7


def test_llama_bench_human_readable_fallback():
    """Non-JSON output falls back to the regex path without crashing."""
    human = (
        "| model | size | test | t/s |\n"
        "| llama | 4B | pp512 | 950.50 tokens per second |\n"
        "| llama | 4B | tg128 | 42.70 tokens per second |\n"
    )
    res = LlamaCppMetalBaseline._parse(human, "m.gguf", n_ctx=512)
    assert res._is_measured is True
    assert res.prefill_tok_per_s == 950.50
    assert res.decode_tok_per_s == 42.70

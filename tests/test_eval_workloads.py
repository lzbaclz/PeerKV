"""CPU tests for the eval foundation: workload generators + serving metrics."""
import math

import pytest

from umallm.eval.workloads import (
    make_long_context, make_multi_turn, synth_text, workload_summary,
)
from umallm.eval.serving_metrics import (
    RequestRecord, deadline_miss_ratio, goodput_tok_s, summarize_run,
    throughput_tok_s, tpot_percentiles,
)


# ---- workloads ---------------------------------------------------------- #
def test_synth_text_deterministic_and_scales():
    a = synth_text(100, seed=1)
    b = synth_text(100, seed=1)
    assert a == b                          # deterministic for a fixed seed
    assert synth_text(0) == ""
    # longer target -> longer text
    assert len(synth_text(400, seed=1)) > len(synth_text(100, seed=1))


def test_long_context_shapes():
    specs = make_long_context(n_requests=8, context_tokens=2048, decode_tokens=64)
    assert len(specs) == 8
    assert all(s.prompt_tokens == 2048 and s.max_new_tokens == 64 for s in specs)
    assert all(s.prompt_text for s in specs)
    assert all(s.arrival_s == 0.0 for s in specs)   # closed-loop by default
    assert len({s.request_id for s in specs}) == 8  # unique ids


def test_long_context_open_loop_arrivals_monotonic():
    specs = make_long_context(n_requests=20, context_tokens=512, decode_tokens=16,
                              arrival_rate_rps=5.0, seed=3)
    arrivals = [s.arrival_s for s in specs]
    assert arrivals == sorted(arrivals)    # non-decreasing
    assert arrivals[-1] > 0.0              # actually spread in time


def test_multi_turn_structure_and_growth():
    specs = make_multi_turn(n_convs=3, n_turns=4, prefix_tokens=1024,
                            turn_tokens=128, decode_tokens=32)
    assert len(specs) == 12
    conv0 = [s for s in specs if s.conversation_id == "conv-0"]
    assert [s.turn for s in conv0] == [0, 1, 2, 3]
    # prompt grows with turn (shared prefix + accumulated dialogue)
    ptoks = [s.prompt_tokens for s in conv0]
    assert ptoks == sorted(ptoks) and ptoks[0] == 1024 and ptoks[-1] == 1024 + 3 * 128


def test_multi_turn_shares_prefix_text():
    specs = make_multi_turn(n_convs=1, n_turns=3, prefix_tokens=512, turn_tokens=64)
    prefixes = [s.prompt_text[:200] for s in specs]
    assert prefixes[0] == prefixes[1] == prefixes[2]  # same shared prefix head


def test_workload_summary():
    specs = make_long_context(n_requests=4, context_tokens=1000, decode_tokens=10)
    s = workload_summary(specs)
    assert s["n"] == 4 and s["prompt_tokens_max"] == 1000
    assert s["total_decode_tokens"] == 40
    assert "long_context" in s["families"]


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        make_long_context(n_requests=0)
    with pytest.raises(ValueError):
        make_multi_turn(n_convs=0)


# ---- serving metrics ---------------------------------------------------- #
def _rec(rid, tpot_s, out=100, ok=True, ttft_s=0.05):
    return RequestRecord(request_id=rid, prompt_tokens=1000, output_tokens=out,
                         ttft_s=ttft_s, tpot_s=tpot_s, start_s=0.0,
                         end_s=ttft_s + tpot_s * out, ok=ok)


def test_tpot_percentiles_interpolation():
    recs = [_rec(f"r{i}", tpot_s=(i + 1) / 1000.0) for i in range(100)]  # 1..100 ms
    p = tpot_percentiles(recs, ps=(50, 99))
    assert abs(p["tpot_p50_ms"] - 50.5) < 1.0    # median ~50.5 ms
    assert p["tpot_p99_ms"] >= p["tpot_p50_ms"]


def test_throughput_and_goodput():
    # two requests, 100 tokens each; one fast (within 50ms), one slow.
    recs = [_rec("fast", tpot_s=0.02), _rec("slow", tpot_s=0.08)]
    assert throughput_tok_s(recs, wall_s=10.0) == pytest.approx(200 / 10.0)
    # deadline 50ms/token -> only the fast one's tokens are goodput
    assert goodput_tok_s(recs, tpot_deadline_s=0.05, wall_s=10.0) == pytest.approx(10.0)


def test_deadline_miss_counts_failures_as_misses():
    recs = [_rec("ok", tpot_s=0.02), _rec("oom", tpot_s=0.0, ok=False)]
    # 1 of 2 missed (the failed one), even though its tpot is 0
    assert deadline_miss_ratio(recs, tpot_deadline_s=0.05) == 0.5


def test_summarize_run_rolls_up():
    recs = [_rec("a", tpot_s=0.03, out=50), _rec("b", tpot_s=0.10, out=50, ok=True)]
    summ = summarize_run(recs, wall_s=5.0, tpot_deadline_s=0.05,
                         extra={"mode": "routeb", "hbm_high_water_gib": 70.1})
    assert summ["n_requests"] == 2 and summ["n_failed"] == 0
    assert summ["mode"] == "routeb" and summ["hbm_high_water_gib"] == 70.1
    assert summ["deadline_miss_ratio"] == 0.5     # b misses 50ms
    assert "tpot_p99_ms" in summ and "throughput_tok_s" in summ
    assert summ["total_output_tokens"] == 100


def test_empty_and_degenerate_inputs():
    assert throughput_tok_s([], 0.0) == 0.0
    assert deadline_miss_ratio([], 0.05) == 0.0
    assert goodput_tok_s([], 0.05, 0.0) == 0.0
    # percentiles of an empty set are nan (not a crash)
    assert math.isnan(tpot_percentiles([])["tpot_p50_ms"])

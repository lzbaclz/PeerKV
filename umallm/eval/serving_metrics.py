"""Serving-time metrics for the Route B benchmark (pure Python, testable).

Given per-request timing records, compute the numbers the paper reports:
TPOT percentiles, throughput, deadline-miss ratio, and goodput (tokens that
met their per-token deadline, per wall-second). All percentile math uses
linear interpolation and is independent of numpy so tests run anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    """Outcome of one issued request.

    ``tpot_s`` is the mean time-per-output-token for this request (s/token);
    ``ttft_s`` the time-to-first-token. ``ok=False`` marks a request that
    failed (e.g. OOM under a too-tight HBM budget) -- counted against the run,
    never silently dropped.
    """

    request_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    tpot_s: float
    start_s: float
    end_s: float
    ok: bool = True
    meta: dict = field(default_factory=dict)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Linear-interpolation percentile of an already-sorted list, p in [0,100]."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def tpot_percentiles(records, ps=(50, 90, 99, 99.9)) -> dict:
    """TPOT percentiles in milliseconds over the OK requests."""
    vals = sorted(r.tpot_s * 1e3 for r in records if r.ok)
    return {f"tpot_p{p:g}_ms": _percentile(vals, p) for p in ps}


def ttft_percentiles(records, ps=(50, 99)) -> dict:
    vals = sorted(r.ttft_s * 1e3 for r in records if r.ok)
    return {f"ttft_p{p:g}_ms": _percentile(vals, p) for p in ps}


def throughput_tok_s(records, wall_s: float) -> float:
    """Output tokens per wall-clock second over the whole run."""
    if wall_s <= 0:
        return 0.0
    return sum(r.output_tokens for r in records if r.ok) / wall_s


def deadline_miss_ratio(records, tpot_deadline_s: float) -> float:
    """Fraction of OK requests whose mean TPOT exceeded the deadline.

    Failed (not-ok) requests count as misses -- an OOM is the hardest miss.
    """
    if not records:
        return 0.0
    misses = sum(1 for r in records if (not r.ok) or r.tpot_s > tpot_deadline_s)
    return misses / len(records)


def goodput_tok_s(records, tpot_deadline_s: float, wall_s: float) -> float:
    """Tokens that met the per-token deadline, per wall-second.

    The serving analogue of throughput-under-SLO: tokens from requests whose
    TPOT was within deadline, divided by wall time. This is the metric that
    rewards keeping hot KV fast *and* fitting more work at once.
    """
    if wall_s <= 0:
        return 0.0
    good = sum(r.output_tokens for r in records
               if r.ok and r.tpot_s <= tpot_deadline_s)
    return good / wall_s


def summarize_run(records, wall_s: float, tpot_deadline_s: float,
                  extra: dict | None = None) -> dict:
    """Roll up one (mode, config) run into the result dict the harness emits."""
    ok = [r for r in records if r.ok]
    out = {
        "n_requests": len(records),
        "n_ok": len(ok),
        "n_failed": len(records) - len(ok),
        "wall_s": wall_s,
        "throughput_tok_s": throughput_tok_s(records, wall_s),
        "goodput_tok_s": goodput_tok_s(records, tpot_deadline_s, wall_s),
        "deadline_miss_ratio": deadline_miss_ratio(records, tpot_deadline_s),
        "tpot_deadline_ms": tpot_deadline_s * 1e3,
        "total_output_tokens": sum(r.output_tokens for r in ok),
        "max_prompt_tokens": max((r.prompt_tokens for r in records), default=0),
    }
    out.update(tpot_percentiles(records))
    out.update(ttft_percentiles(records))
    if extra:
        out.update(extra)
    return out

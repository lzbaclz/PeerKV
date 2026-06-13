"""Workload generation for the Route B GH200 benchmarks.

Pure Python (no torch/vLLM/CUDA) so it is unit-testable anywhere. Produces
``RequestSpec`` lists the harness feeds to a backend. Two families that stress
KV-cache residency differently:

  * **long_context** -- N independent requests, each a long prompt + a decode
    tail. Stresses *capacity* (does the KV fit at all?) and cold-block
    placement: after prefill, most of a long prompt's KV goes cold, so a good
    residency policy demotes it to Grace and keeps only sink+window hot.
  * **multi_turn** -- M conversations of T turns sharing a long system prefix.
    Stresses *reuse*: the shared prefix's KV should be demoted between turns
    and restored on the next turn (the Part 1 external-reuse path), instead of
    recomputed.

Token counts are *targets*. ``synth_text`` emits deterministic filler whose
length tracks a ~4-chars/token heuristic; the harness re-tokenizes with the
model's real tokenizer and trims/pads to the exact target when running for
real. Determinism is via an explicit seed so a benchmark is reproducible.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

# A small fixed vocabulary keeps generated prompts deterministic and
# tokenizer-friendly (common English words rarely split into many subwords).
_WORDS = (
    "the system processes a request and writes its attention state into the "
    "key value cache which then must be kept resident or moved to a slower "
    "tier when memory pressure rises so that long context generation can "
    "continue without exceeding the available high bandwidth memory budget "
    "while still meeting the per token latency target that the operator set "
    "for this workload under a service level objective"
).split()

_APPROX_CHARS_PER_TOKEN = 4


@dataclass
class RequestSpec:
    """One request the harness will issue.

    ``prompt_tokens`` / ``max_new_tokens`` are targets; ``prompt_text`` is
    deterministic filler the harness may re-tokenize. ``arrival_s`` supports
    open-loop arrival; ``conversation_id``/``turn`` tag multi-turn requests so
    a backend can reuse the shared prefix.
    """

    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_s: float = 0.0
    conversation_id: str | None = None
    turn: int = 0
    prompt_text: str | None = None
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def synth_text(n_tokens: int, seed: int = 0) -> str:
    """Deterministic filler text of roughly ``n_tokens`` tokens.

    Draws from a fixed word list with a seeded RNG. The harness re-tokenizes
    with the real tokenizer and trims to the exact count; this only needs to
    be *long enough* and reproducible.
    """
    if n_tokens <= 0:
        return ""
    rng = random.Random(seed)
    # Over-generate ~15% to give the tokenizer room to trim down to target.
    target_words = int(n_tokens * 1.15) + 4
    words = [rng.choice(_WORDS) for _ in range(target_words)]
    return " ".join(words)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


def make_long_context(
    n_requests: int = 32,
    context_tokens: int = 8192,
    decode_tokens: int = 256,
    seed: int = 0,
    arrival_rate_rps: float | None = None,
) -> list[RequestSpec]:
    """N independent long-context requests.

    ``arrival_rate_rps`` None -> all arrive at t=0 (closed-loop/batched);
    otherwise Poisson-ish open-loop arrivals at the given rate.
    """
    if n_requests <= 0:
        raise ValueError("n_requests must be > 0")
    if context_tokens <= 0 or decode_tokens <= 0:
        raise ValueError("context_tokens and decode_tokens must be > 0")
    rng = random.Random(seed)
    specs: list[RequestSpec] = []
    t = 0.0
    for i in range(n_requests):
        if arrival_rate_rps:
            # exponential inter-arrival for an open-loop Poisson process
            t += rng.expovariate(arrival_rate_rps)
        specs.append(RequestSpec(
            request_id=f"lc-{i}",
            prompt_tokens=context_tokens,
            max_new_tokens=decode_tokens,
            arrival_s=t,
            prompt_text=synth_text(context_tokens, seed=seed + i),
            meta={"family": "long_context"},
        ))
    return specs


def make_multi_turn(
    n_convs: int = 16,
    n_turns: int = 4,
    prefix_tokens: int = 4096,
    turn_tokens: int = 256,
    decode_tokens: int = 128,
    seed: int = 0,
) -> list[RequestSpec]:
    """M conversations x T turns sharing a long per-conversation prefix.

    Turn ``k`` of a conversation has a prompt of ``prefix_tokens + k*turn_tokens``
    (the shared prefix plus the accumulated dialogue). Requests are ordered so
    a conversation's turns are contiguous, which is what exercises demote-on-
    idle then restore-on-next-turn for the shared prefix.
    """
    if n_convs <= 0 or n_turns <= 0:
        raise ValueError("n_convs and n_turns must be > 0")
    if prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be > 0")
    specs: list[RequestSpec] = []
    for c in range(n_convs):
        prefix = synth_text(prefix_tokens, seed=seed + c)
        for k in range(n_turns):
            ptoks = prefix_tokens + k * turn_tokens
            tail = synth_text(k * turn_tokens, seed=seed + 1000 * c + k) if k else ""
            specs.append(RequestSpec(
                request_id=f"mt-{c}-{k}",
                prompt_tokens=ptoks,
                max_new_tokens=decode_tokens,
                conversation_id=f"conv-{c}",
                turn=k,
                prompt_text=(prefix + " " + tail).strip(),
                meta={"family": "multi_turn", "prefix_tokens": prefix_tokens},
            ))
    return specs


def workload_summary(specs: list[RequestSpec]) -> dict:
    """Quick descriptive stats for a workload (for logging / the result JSON)."""
    if not specs:
        return {"n": 0}
    ptoks = [s.prompt_tokens for s in specs]
    ntoks = [s.max_new_tokens for s in specs]
    convs = {s.conversation_id for s in specs if s.conversation_id}
    return {
        "n": len(specs),
        "n_conversations": len(convs),
        "prompt_tokens_min": min(ptoks),
        "prompt_tokens_max": max(ptoks),
        "prompt_tokens_mean": sum(ptoks) / len(ptoks),
        "total_prompt_tokens": sum(ptoks),
        "total_decode_tokens": sum(ntoks),
        "families": sorted({s.meta.get("family", "?") for s in specs}),
    }

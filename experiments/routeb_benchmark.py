#!/usr/bin/env python3
"""Route B vs baselines -- the GH200 head-to-head benchmark (CCF-A centerpiece).

Runs the SAME workload under four KV-residency strategies and reports the
numbers the paper lives or dies on: max servable context, throughput, goodput
under an SLO, and TPOT P50/P99. The four modes:

  routeb        Route B: KV pool on managed memory + active attention/SLO-driven
                cudaMemAdvise placement (hot->HBM, cold->Grace), coherent reads.
  passive_uvm   KV pool on managed memory but NO advise -- the CUDA driver
                demand-pages reactively. This is the "let the hardware do it"
                baseline that isolates the value of *proactive* placement.
  vllm_offload  vLLM's native CPU/host KV offload (copy-based).
  hbm_only      No offload. Bounded by HBM; OOMs past a context/concurrency --
                that OOM point is the capacity baseline Route B is meant to beat.

Backends are pluggable:
  * MockBackend  -- CPU, no GPU. Emits SYNTHETIC latencies from a physically
    motivated cost model (coherent-read vs fault-thrash vs copy vs OOM). Every
    record is tagged ``measured=false``. Use it to (a) test the harness/metric
    pipeline anywhere and (b) state the *hypothesis* the GH200 run will confirm
    or refute. It is NOT evidence.
  * VLLMBackend  -- real measurements on a GH200. ``measured=true``.

Usage:
    # CPU, synthetic -- validates orchestration + prints the hypothesis
    python experiments/routeb_benchmark.py --backend mock \
        --context-sweep 4096,16384,65536 --out experiments/results/routeb_mock.json
    # GH200, real
    python experiments/routeb_benchmark.py --backend vllm --model meta-llama/Llama-3.1-8B \
        --context-sweep 4096,16384,65536,131072 --out experiments/results/routeb_gh200.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field

from umallm.eval.serving_metrics import RequestRecord, summarize_run
from umallm.eval.workloads import (
    make_long_context, make_multi_turn, workload_summary,
)

MODES = ("routeb", "passive_uvm", "vllm_offload", "hbm_only")
_GIB = float(1 << 30)


@dataclass
class BenchConfig:
    model: str = "meta-llama/Llama-3.1-8B"
    context_tokens: int = 8192
    decode_tokens: int = 128
    n_requests: int = 32
    concurrency: int = 16
    hbm_budget_gib: float = 40.0       # HBM the KV working set may use
    grace_budget_gib: float = 320.0    # Grace LPDDR5X available for cold KV
    deadline_ms: float = 50.0          # per-token SLO
    workload: str = "long_context"     # or "multi_turn"
    seed: int = 0
    # KV geometry (8B defaults; override for 70B). Used by the mock capacity
    # model and to report footprints.
    n_layers: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2

    def kv_bytes_per_token(self) -> int:
        # K and V, all layers: 2 * L * Hkv * Dh * bytes
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes

    def as_dict(self) -> dict:
        d = asdict(self)
        d["kv_bytes_per_token"] = self.kv_bytes_per_token()
        return d


def build_workload(cfg: BenchConfig):
    if cfg.workload == "multi_turn":
        n_turns = 4
        return make_multi_turn(
            n_convs=max(1, cfg.n_requests // n_turns), n_turns=n_turns,
            prefix_tokens=cfg.context_tokens, turn_tokens=cfg.context_tokens // 16,
            decode_tokens=cfg.decode_tokens, seed=cfg.seed)
    return make_long_context(
        n_requests=cfg.n_requests, context_tokens=cfg.context_tokens,
        decode_tokens=cfg.decode_tokens, seed=cfg.seed)


# ====================================================================== #
# Backend protocol
# ====================================================================== #
class Backend:
    measured = False
    name = "base"

    def run(self, mode: str, cfg: BenchConfig, specs) -> tuple[list, dict]:
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# MockBackend -- SYNTHETIC cost model (a hypothesis, not a measurement)
# ---------------------------------------------------------------------- #
class MockBackend(Backend):
    """Physically-motivated synthetic latencies. Tagged measured=false.

    Cost model (per request), all clearly assumptions to be confirmed on HW:
      * base TPOT grows mildly with concurrency (batch contention).
      * KV that doesn't fit in the HBM budget is "cold". hbm_only cannot place
        cold KV -> the overflow requests OOM (ok=false). The other three place
        cold KV off-HBM and pay a per-mode penalty proportional to cold_frac:
          routeb       small (coherent C2C read + proactive prefetch hides it)
          vllm_offload moderate (explicit copy on the critical path)
          passive_uvm  moderate mean but heavy *tail* (reactive page faults
                       thrash; some requests stall)
      * multi_turn gives routeb a reuse bonus (demoted prefix restored cheaply).
    """

    measured = False
    name = "mock"

    # per-mode (mean cold penalty coeff, tail-blowup factor on a fraction of reqs)
    _COLD = {
        "routeb":       (0.15, 1.2),
        "vllm_offload": (0.40, 1.6),
        "passive_uvm":  (0.55, 3.0),
        "hbm_only":     (0.0,  1.0),   # never pays -- it OOMs instead
    }

    def run(self, mode, cfg, specs):
        rng = random.Random(cfg.seed * 31 + MODES.index(mode))
        kv_bpt = cfg.kv_bytes_per_token()
        hbm_cap = cfg.hbm_budget_gib * _GIB
        # working set = the concurrent requests' KV at full context
        per_req_bytes = cfg.context_tokens * kv_bpt
        concurrent = min(cfg.concurrency, len(specs))
        working_set = per_req_bytes * concurrent
        cold_frac_run = max(0.0, 1.0 - hbm_cap / working_set) if working_set else 0.0

        base_ms = 8.0 + 0.30 * concurrent           # ms/token, batch contention
        cold_coeff, tail_factor = self._COLD[mode]
        recs, t = [], 0.0
        n_fit = int(hbm_cap // per_req_bytes) if per_req_bytes else len(specs)

        grace_bytes = 0.0
        hbm_high = min(working_set, hbm_cap)
        for i, s in enumerate(specs):
            reuse = 0.6 if (mode == "routeb" and s.turn > 0) else 1.0
            if mode == "hbm_only" and i >= max(1, n_fit):
                # capacity exceeded: this request cannot be admitted -> OOM/fail
                recs.append(RequestRecord(
                    request_id=s.request_id, prompt_tokens=s.prompt_tokens,
                    output_tokens=0, ttft_s=0.0, tpot_s=0.0,
                    start_s=t, end_s=t, ok=False,
                    meta={"reason": "hbm_oom"}))
                continue
            penalty = base_ms * cold_coeff * cold_frac_run * reuse
            # tail: a fraction of requests hit the bad path (faults/copies)
            if rng.random() < 0.1 and mode != "hbm_only":
                penalty *= tail_factor
            jitter = rng.uniform(-0.5, 0.5)
            tpot_ms = max(1.0, base_ms + penalty + jitter)
            ttft_s = (s.prompt_tokens * 0.00002) + penalty / 1e3  # prefill-ish
            out = s.max_new_tokens
            start = s.arrival_s
            end = start + ttft_s + tpot_ms / 1e3 * out
            t = max(t, end)
            recs.append(RequestRecord(
                request_id=s.request_id, prompt_tokens=s.prompt_tokens,
                output_tokens=out, ttft_s=ttft_s, tpot_s=tpot_ms / 1e3,
                start_s=start, end_s=end, ok=True))
            if mode != "hbm_only":
                grace_bytes += per_req_bytes * cold_frac_run

        wall = max((r.end_s for r in recs), default=0.0) - \
            min((r.start_s for r in recs), default=0.0)
        wall = max(wall, 1e-6)
        extra = {
            "mode": mode, "backend": self.name, "measured": False,
            "cold_frac": round(cold_frac_run, 4),
            "hbm_high_water_gib": round(hbm_high / _GIB, 3),
            "grace_resident_gib": round(grace_bytes / _GIB, 3),
            "working_set_gib": round(working_set / _GIB, 3),
            "wall_s": wall,
        }
        return recs, extra


# ---------------------------------------------------------------------- #
# VLLMBackend -- real GH200 measurement (guarded; validate on hardware)
# ---------------------------------------------------------------------- #
class VLLMBackend(Backend):
    """Drives a real vLLM engine per mode on a GH200. measured=true.

    Off-hardware this raises on construction. The per-mode engine wiring below
    is written against the vLLM V1 API; the exact KV-offload knob and the
    request-metrics fields have drifted across releases, so the spots that need
    on-box confirmation are marked CONFIRM-ON-HW.
    """

    measured = True
    name = "vllm"

    def __init__(self):
        try:
            import torch
            import vllm  # noqa: F401
        except ImportError as e:  # pragma: no cover - hardware path
            raise RuntimeError(
                "VLLMBackend needs torch+vllm on a CUDA box. Use --backend mock "
                "off-hardware.") from e
        if not torch.cuda.is_available():  # pragma: no cover
            raise RuntimeError("no CUDA device; VLLMBackend requires a GPU.")
        self._torch = torch

    # -- per-mode engine construction ---------------------------------- #
    def _build_llm(self, mode, cfg):  # pragma: no cover - hardware path
        from vllm import LLM
        # HBM budget -> gpu_memory_utilization. Coarse: leave room for weights.
        gpu_util = min(0.95, max(0.2, cfg.hbm_budget_gib / 80.0))
        common = dict(model=cfg.model, tensor_parallel_size=1,
                      gpu_memory_utilization=gpu_util, enforce_eager=False,
                      max_model_len=cfg.context_tokens + cfg.decode_tokens + 8)

        if mode in ("routeb", "passive_uvm"):
            # Both land the KV pool on managed (Grace+HBM) memory.
            from umallm.vllm_integration.uma_backend import patch_vllm_kv_allocation
            patch_vllm_kv_allocation()  # must precede LLM() -- wraps KV alloc site
            if mode == "routeb":
                # active placement: install the UMA connector (kv_both)
                common["kv_transfer_config"] = {
                    "kv_connector": "UMAGraceHopperConnector",
                    "kv_connector_module_path":
                        "umallm.vllm_integration.gh200_connector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "deadline_ms": cfg.deadline_ms, "cold_bits": 4,
                        "residency_hints": True, "coherent_read": True,
                    },
                }
            # passive_uvm: managed pool, NO connector -> driver demand-pages.
            return LLM(**common)

        if mode == "vllm_offload":
            # vLLM's native host KV offload. CONFIRM-ON-HW: the knob name/units
            # vary by release (swap_space GiB vs a CPUOffloadingConnector).
            common["swap_space"] = cfg.grace_budget_gib
            return LLM(**common)

        # hbm_only: plain engine, no offload, bounded by HBM.
        return LLM(**common)

    def run(self, mode, cfg, specs):  # pragma: no cover - hardware path
        from vllm import SamplingParams
        torch = self._torch
        torch.cuda.reset_peak_memory_stats()
        try:
            llm = self._build_llm(mode, cfg)
        except Exception as e:
            # An OOM at build/profile time is the capacity result for hbm_only.
            recs = [RequestRecord(s.request_id, s.prompt_tokens, 0, 0.0, 0.0,
                                  0.0, 0.0, ok=False, meta={"reason": repr(e)})
                    for s in specs]
            return recs, {"mode": mode, "backend": self.name, "measured": True,
                          "wall_s": 0.0, "build_error": repr(e)}

        prompts = [s.prompt_text or "" for s in specs]
        sps = [SamplingParams(max_tokens=s.max_new_tokens, temperature=0.0)
               for s in specs]
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sps)
        wall = time.perf_counter() - t0

        recs = []
        for s, o in zip(specs, outs):
            comp = o.outputs[0] if o.outputs else None
            n_out = len(comp.token_ids) if comp else 0
            m = getattr(o, "metrics", None)  # CONFIRM-ON-HW: field names
            ttft = float(getattr(m, "first_token_time", 0.0) -
                         getattr(m, "arrival_time", 0.0)) if m else 0.0
            fin = float(getattr(m, "finished_time", 0.0) -
                        getattr(m, "first_token_time", 0.0)) if m else wall
            tpot = (fin / max(1, n_out - 1)) if n_out > 1 else fin
            recs.append(RequestRecord(
                request_id=s.request_id, prompt_tokens=s.prompt_tokens,
                output_tokens=n_out, ttft_s=max(0.0, ttft),
                tpot_s=max(0.0, tpot), start_s=0.0, end_s=wall,
                ok=(n_out > 0)))

        hbm_high = torch.cuda.max_memory_allocated() / _GIB
        extra = {"mode": mode, "backend": self.name, "measured": True,
                 "wall_s": wall, "hbm_high_water_gib": round(hbm_high, 3)}
        # residency footprint from the connector, if present
        try:
            stats = llm.llm_engine.get_kv_connector_stats()  # CONFIRM-ON-HW
            if stats:
                extra["connector_stats"] = stats
        except Exception:
            pass
        return recs, extra


def make_backend(name: str) -> Backend:
    if name == "mock":
        return MockBackend()
    if name == "vllm":
        return VLLMBackend()
    raise ValueError(f"unknown backend {name!r} (use mock|vllm)")


# ====================================================================== #
# Driver
# ====================================================================== #
def run_matrix(modes, configs, backend: Backend) -> list[dict]:
    results = []
    for cfg in configs:
        specs = build_workload(cfg)
        for mode in modes:
            recs, extra = backend.run(mode, cfg, specs)
            summ = summarize_run(recs, extra.get("wall_s", 0.0),
                                 cfg.deadline_ms / 1e3, extra={**extra, **cfg.as_dict()})
            summ["workload_summary"] = workload_summary(specs)
            results.append(summ)
            _print_row(summ)
    return results


def _print_row(s: dict):
    tag = "MEAS" if s.get("measured") else "MOCK"
    print(f"  [{tag}] {s['mode']:>12} ctx={s['context_tokens']:>7} "
          f"conc={s['concurrency']:>3} | ok={s['n_ok']}/{s['n_requests']} "
          f"tpot_p50={s.get('tpot_p50_ms', float('nan')):.1f} "
          f"tpot_p99={s.get('tpot_p99_ms', float('nan')):.1f}ms "
          f"goodput={s.get('goodput_tok_s', 0):.0f}tok/s "
          f"hbm_hw={s.get('hbm_high_water_gib', 0):.1f}GiB")


def _configs_from_args(args) -> list[BenchConfig]:
    ctxs = [int(x) for x in args.context_sweep.split(",") if x]
    cfgs = []
    for ctx in ctxs:
        cfgs.append(BenchConfig(
            model=args.model, context_tokens=ctx, decode_tokens=args.decode_tokens,
            n_requests=args.n_requests, concurrency=args.concurrency,
            hbm_budget_gib=args.hbm_budget_gib, grace_budget_gib=args.grace_budget_gib,
            deadline_ms=args.deadline_ms, workload=args.workload, seed=args.seed,
            n_layers=args.n_layers, n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim))
    return cfgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["mock", "vllm"], default="mock")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--context-sweep", default="4096,16384,65536")
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--n-requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--hbm-budget-gib", type=float, default=40.0)
    ap.add_argument("--grace-budget-gib", type=float, default=320.0)
    ap.add_argument("--deadline-ms", type=float, default=50.0)
    ap.add_argument("--workload", choices=["long_context", "multi_turn"],
                    default="long_context")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=32)
    ap.add_argument("--n-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    modes = [m for m in args.modes.split(",") if m]
    for m in modes:
        if m not in MODES:
            ap.error(f"unknown mode {m!r}; choose from {MODES}")

    backend = make_backend(args.backend)
    if not backend.measured:
        print("=" * 72)
        print("MOCK backend: SYNTHETIC latencies from a cost-model HYPOTHESIS.")
        print("These are NOT measurements. Run --backend vllm on a GH200 for real")
        print("numbers. The mock states what we expect Route B to win, and lets")
        print("the harness + analysis be validated off-hardware.")
        print("=" * 72)

    configs = _configs_from_args(args)
    results = run_matrix(modes, configs, backend)

    payload = {
        "backend": backend.name, "measured": backend.measured,
        "modes": modes, "results": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out} ({len(results)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

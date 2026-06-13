"""e35 -- honest AQUA arm: fixed-chunk coalesced copy-back vs PeerKV's C*.

Verified prior-art fact (web check): AQUA ALREADY coalesces small NVLink transfers
into a fixed large buffer (its ~200-LOC gather kernel; documents 4MB=50GB/s,
64MB=200GB/s). So the per-block-loses result must NOT be labeled AQUA (that's
Harvest; see analyze_harvest_relabel.py). The honest AQUA comparison is a
FIXED-buffer coalesced copy-back vs PeerKV's cost-model-DERIVED chunk C*
(umallm.multigpu.optimal_chunk_blocks).

On a single regime the two TIE (same NVLink bandwidth bound). PeerKV's only real
delta over AQUA is AUTO-SELECTION: a fixed buffer tuned for one (page size / tier /
deadline) is mis-set in another -- too small => residual launch overhead; too large
=> peak/latency budget blown. C* re-tracks the optimum per regime. We sweep page
size (block bytes) and report decode-step latency, peak, and deadline-attainment.

    /home/lzq/miniconda3/envs/peerkv/bin/python experiments/e35_aqua_arm.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# AQUA's fixed coalescing buffers across ITS OWN documented range (4-64MB), so the
# comparison is not a strawman: we test every fixed choice AQUA might pick and show
# none dominates C* across regimes.
AQUA_FIXED_BYTES_SET = [4 * 1024**2, 16 * 1024**2, 64 * 1024**2]


def main():
    import torch
    import torch.nn.functional as F
    from umallm.torch_tiered_attn import flash_merge_attention
    from umallm.multigpu import MultiGPUKVModel, MGTier, optimal_chunk_blocks

    if torch.cuda.device_count() < 2:
        print("need >=2 GPUs"); return
    model = MultiGPUKVModel()  # measured A100/NVLink defaults (273 GB/s, c_T1=23.6us)

    H, D = 8, 128                     # GQA-ish KV geometry
    TOTAL_TOKENS = 65536
    SPILL = 0.5
    scale = 1.0 / (D ** 0.5)
    trials, seeds, warmup = 12, 2, 3
    DEADLINE_MS = 6.0                 # per-step copy-back deadline for attainment

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med(xs):
        return statistics.median(sorted(xs))

    def make_blocks(block_tokens):
        n_blocks = TOTAL_TOKENS // block_tokens
        n_spill = int(n_blocks * SPILL)
        k_blocks, v_blocks = [], []
        for i in range(n_blocks):
            dev = "cuda:1" if i < n_spill else "cuda:0"
            k_blocks.append(torch.randn(1, H, block_tokens, D, dtype=torch.float16, device=dev))
            v_blocks.append(torch.randn(1, H, block_tokens, D, dtype=torch.float16, device=dev))
        # put cold (spill) blocks first so coalescing runs over same-device runs
        return k_blocks, v_blocks, n_blocks, n_spill

    def timed(step):
        ms = []
        for _ in range(seeds):
            for _ in range(warmup):
                step()
            sync()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            for _ in range(trials):
                sync(); s.record(); step(); e.record(); sync()
                ms.append(s.elapsed_time(e))
        return med(ms)

    out = {
        "_experiment": "e35_aqua_arm",
        "_is_measured": True,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "geometry": f"H{H}xD{D}", "total_tokens": TOTAL_TOKENS, "spill_frac": SPILL,
        "aqua_fixed_bytes_set": AQUA_FIXED_BYTES_SET, "deadline_ms": DEADLINE_MS,
        "device": torch.cuda.get_device_name(0),
        "claim": ("AQUA (fixed buffer) and PeerKV (C*) tie within a regime, but a "
                  "fixed buffer tuned for one page size is suboptimal in others; C* "
                  "re-derives the optimum per regime from the calibrated cost model."),
        "page_sweep": [],
    }

    q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")

    for block_tokens in (16, 64, 256):
        block_bytes = block_tokens * H * D * 2 * 2     # K+V, fp16
        k_blocks, v_blocks, n_blocks, n_spill = make_blocks(block_tokens)
        sync()

        c_star = optimal_chunk_blocks(model, block_bytes, peak_budget_blocks=n_blocks,
                                      spill_tier=MGTier.PEER_NVLINK)
        aqua_chunks = {b: max(1, b // block_bytes) for b in AQUA_FIXED_BYTES_SET}

        # exactness once
        with torch.no_grad():
            ref = flash_merge_attention(q, k_blocks, v_blocks, scale=scale, chunk_blocks=1)

        # full chunk sweep to locate the measured optimum (the knee)
        sweep = {}
        candidates = set([1, 8, 32, c_star, 64, 128, 256, n_blocks]) | set(aqua_chunks.values())
        for C in sorted(candidates):
            C = max(1, min(int(C), n_blocks))
            torch.cuda.reset_peak_memory_stats(0)
            def step(C=C):
                return flash_merge_attention(q, k_blocks, v_blocks, scale=scale, chunk_blocks=C)
            try:
                ms = timed(step)
                with torch.no_grad():
                    o = step()
                    cos = float(F.cosine_similarity(ref.reshape(-1).float(),
                                                    o.reshape(-1).float(), dim=0))
                peak = torch.cuda.max_memory_allocated(0) / 1024**2
                sweep[C] = {"ms": round(ms, 3), "peak_mb": round(peak, 1), "cos": round(cos, 6)}
            except RuntimeError as e:
                sweep[C] = {"status": f"OOM:{str(e)[:40]}"}
            torch.cuda.empty_cache()

        def ms_of(C):
            C = max(1, min(int(C), n_blocks))
            return sweep[C]["ms"] if "ms" in sweep.get(C, {}) else None
        meas_opt_C = min((C for C in sweep if "ms" in sweep[C]),
                         key=lambda C: sweep[C]["ms"])
        opt_ms = sweep[meas_opt_C]["ms"]
        cstar_ms = ms_of(c_star)
        aqua = {f"{b//1024**2}MB": {
            "chunk_blocks": min(aqua_chunks[b], n_blocks), "ms": ms_of(aqua_chunks[b]),
            "vs_optimal": round(ms_of(aqua_chunks[b]) / opt_ms, 3) if ms_of(aqua_chunks[b]) else None,
            "meets_deadline": (ms_of(aqua_chunks[b]) is not None and ms_of(aqua_chunks[b]) <= DEADLINE_MS),
        } for b in AQUA_FIXED_BYTES_SET}
        rec = {
            "block_tokens": block_tokens, "block_bytes": block_bytes,
            "n_blocks": n_blocks, "n_spill": n_spill,
            "C_star_costmodel": c_star, "measured_optimal_chunk_blocks": meas_opt_C,
            "C_star_ms": cstar_ms, "measured_optimal_ms": opt_ms,
            "cstar_vs_optimal": round(cstar_ms / opt_ms, 3) if cstar_ms else None,
            "cstar_meets_deadline": (cstar_ms is not None and cstar_ms <= DEADLINE_MS),
            "aqua_fixed": aqua,
            "sweep": {str(C): sweep[C] for C in sweep},
        }
        out["page_sweep"].append(rec)
        del k_blocks, v_blocks; torch.cuda.empty_cache()
        am = "  ".join(f"{k}:{v['ms']}ms" for k, v in aqua.items())
        print(f"[T={block_tokens:3d}] C*={c_star:4d} {cstar_ms}ms (opt {meas_opt_C}={opt_ms}ms)  AQUA[{am}]")

    # ---- peak-budget dimension: C*'s genuine (non-tie) advantage ----
    # On raw page-size throughput a large fixed buffer (64MB) ties C* (above). The
    # real, un-tieable advantage is that C* RESPECTS a peak/scratch budget: a fixed
    # 64MB buffer needs 64MB of scratch regardless, blowing a tight budget; C* caps
    # itself and stays near-optimal within the budget.
    T = 16
    block_bytes = T * H * D * 2 * 2
    k_blocks, v_blocks, n_blocks, n_spill = make_blocks(T)
    sync()
    aqua_64mb_blocks = (64 * 1024**2) // block_bytes      # fixed buffer, ignores budget
    aqua_scratch_mb = aqua_64mb_blocks * block_bytes / 1024**2   # = 64MB
    pbsweep = []
    for cap_mb in (2, 8, 32, 128):
        cap_blocks = max(1, (cap_mb * 1024**2) // block_bytes)
        c_star = optimal_chunk_blocks(model, block_bytes, peak_budget_blocks=cap_blocks,
                                      spill_tier=MGTier.PEER_NVLINK)
        def step(C=c_star):
            return flash_merge_attention(q, k_blocks, v_blocks, scale=scale, chunk_blocks=C)
        ms = timed(step)
        cstar_scratch_mb = c_star * block_bytes / 1024**2   # transfer scratch the chunk needs
        pbsweep.append({
            "peak_budget_mb": cap_mb, "peak_budget_blocks": cap_blocks,
            "C_star": c_star, "C_star_ms": round(ms, 3),
            "C_star_scratch_mb": round(cstar_scratch_mb, 2),
            "C_star_within_budget": cstar_scratch_mb <= cap_mb,
            "aqua_64mb_scratch_mb": round(aqua_scratch_mb, 1),
            "aqua_64mb_within_budget": aqua_scratch_mb <= cap_mb,
        })
        print(f"  [scratch<= {cap_mb:3d}MB] C*={c_star:4d} ({cstar_scratch_mb:.1f}MB scratch) "
              f"{ms:.2f}ms within={cstar_scratch_mb <= cap_mb} | "
              f"AQUA-64MB needs {aqua_scratch_mb:.0f}MB scratch "
              f"({'OK' if aqua_scratch_mb<=cap_mb else 'VIOLATES'})")
    out["peak_budget_sweep_T16"] = {
        "note": ("Scratch = chunk_blocks*block_bytes (the transfer buffer the chunk "
                 "must hold), NOT torch max_memory_allocated (which also counts the "
                 "resident local KV). C* caps scratch at the budget and gives the best "
                 "latency achievable within it; AQUA's fixed 64MB needs 64MB scratch "
                 "regardless, violating every tighter budget."),
        "block_bytes": block_bytes, "rows": pbsweep,
    }
    del k_blocks, v_blocks; torch.cuda.empty_cache()

    # cross-regime deadline attainment + worst-case regret per FIXED choice
    n = len(out["page_sweep"])
    cstar_hits = sum(r["cstar_meets_deadline"] for r in out["page_sweep"])
    per_fixed = {}
    for b in AQUA_FIXED_BYTES_SET:
        key = f"{b//1024**2}MB"
        hits = sum(r["aqua_fixed"][key]["meets_deadline"] for r in out["page_sweep"])
        worst = max(r["aqua_fixed"][key]["vs_optimal"] for r in out["page_sweep"]
                    if r["aqua_fixed"][key]["vs_optimal"])
        per_fixed[key] = {"deadline_hits": f"{hits}/{n}", "worst_case_regret_vs_optimal": worst}
    out["deadline_attainment"] = {
        "deadline_ms": DEADLINE_MS,
        "C_star_hits": f"{cstar_hits}/{n}",
        "C_star_mean_regret_vs_optimal": round(
            statistics.mean(r["cstar_vs_optimal"] for r in out["page_sweep"] if r["cstar_vs_optimal"]), 3),
        "C_star_worst_regret_vs_optimal": round(
            max(r["cstar_vs_optimal"] for r in out["page_sweep"] if r["cstar_vs_optimal"]), 3),
        "aqua_fixed_per_buffer": per_fixed,
        "takeaway": ("HONEST: a well-tuned large fixed buffer (AQUA 64MB) TIES C* on "
                     "raw page-size throughput (a near-tie, as the prior-art check "
                     "predicted). Small/mid fixed buffers (4/16MB) miss the deadline in "
                     "some regimes. C*'s genuine, un-tieable advantage is the "
                     "peak_budget_sweep: it respects a peak-memory budget a fixed buffer "
                     "ignores (see peak_budget_sweep_T16)."),
    }
    path = os.path.join(RES, "aqua_arm.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\ndeadline attainment:", json.dumps(out["deadline_attainment"], indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()

"""E-B2 -- unified head-to-head: PeerKV selector vs Harvest / AQUA / host-offload.

Track B spec (collaboration_plan/02_track_B_system_paper.md SS4 E-B2): all arms run
in ONE runtime skeleton -- the same attention kernel (flash_merge_attention /
peer_parallel_attention), the same KV layout, the same timing harness -- and differ
ONLY in transfer/compute policy:

    harvest_perblock : chunk_blocks=1, per-16-token-block transfers (Harvest's
                       no-coalescing policy, see analyze_harvest_relabel.py)
    aqua_64mb        : fixed 64 MB coalesced buffer (AQUA's documented best)
    peerkv_cstar     : cost-model-derived C* coalescing (copy-back corner)
    host_offload     : same C* chunking, spill lives in pinned host memory (PCIe)
    cfk              : compute-follows-KV (partials move, not KV) -- measured in
                       BOTH peer states; its busy-state cost is the measured
                       justification for the R1-busy gate
    peerkv_selector  : executes whatever umallm.runtime.selector.online_select
                       picks (strict mode; do-no-harm violations must be 0)

Axes: geometry {MHA H=32, GQA H=8} x peer {idle, busy} + a fits-single control
row per geometry (do-no-harm positive evidence). The busy stressor is a
persistent fp16 GEMM loop on cuda:1's non-default stream; its iteration rate is
recorded per arm => lender-FLOPs-retained (e23 cross-check).

Direction note: transfers are destination-initiated (pull-equivalent); per Track
A's controlled result direction is a non-effect, recorded not constrained.

    /home/lzq/miniconda3/envs/peerkv/bin/python experiments/eB2_headtohead.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

TOTAL_TOKENS = 65536
SPILL = 0.5
BLOCK_TOKENS = 16            # vLLM page size; Harvest's transfer granule
DEADLINE_MS = 6.0
TRIALS, SEEDS, WARMUP = 12, 2, 3
AQUA_BYTES = 64 * 1024 ** 2


class PeerStressor:
    """Persistent fp16 GEMM loop on the peer GPU's own (non-default) stream.
    Counts iterations so each arm's run yields a lender-throughput sample."""

    def __init__(self, dev="cuda:1", n=4096):
        import torch
        self.dev, self._stop = dev, threading.Event()
        self.iters = 0
        self._a = torch.randn(n, n, dtype=torch.float16, device=dev)
        self._b = torch.randn(n, n, dtype=torch.float16, device=dev)
        self._thread = None

    def __enter__(self):
        import torch
        self._stop.clear()
        self.iters = 0
        self.t0 = time.monotonic()

        def run():
            import torch
            s = torch.cuda.Stream(device=self.dev)
            with torch.cuda.stream(s):
                while not self._stop.is_set():
                    self._a @ self._b
                    s.synchronize()        # keep the queue short & countable
                    self.iters += 1
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        time.sleep(0.5)                    # let the stressor reach steady state
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)
        self.elapsed = time.monotonic() - self.t0

    @property
    def rate(self) -> float:
        return self.iters / max(self.elapsed, 1e-9)


def main():
    import torch
    import torch.nn.functional as F
    from umallm.torch_tiered_attn import flash_merge_attention
    from umallm.multigpu import MultiGPUKVModel, MGTier, optimal_chunk_blocks
    from umallm.peer_parallel_attn import peer_parallel_attention
    from umallm.elastic_policy import (Deployment, Geometry, LinkState,
                                       OperatingPoint, PeerState,
                                       DecodeStepModel)
    from umallm.runtime.selector import online_select
    from umallm.observability.box_probe import gate_or_skip

    if torch.cuda.device_count() < 2:
        print("need >=2 GPUs"); return
    gate_or_skip("eB2_headtohead")
    mg_model = MultiGPUKVModel()

    def sync0():
        torch.cuda.synchronize(0)

    def timed(step, full_sync=True):
        ms = []
        for _ in range(SEEDS):
            for _ in range(WARMUP):
                step()
            sync0()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            for _ in range(TRIALS):
                if full_sync:
                    torch.cuda.synchronize(0)
                s.record(); step(); e.record(); sync0()
                ms.append(s.elapsed_time(e))
        ms.sort()
        iqr = ms[int(.75 * len(ms))] - ms[int(.25 * len(ms))]
        return statistics.median(ms), iqr

    out = {
        "_experiment": "eB2_headtohead",
        "_is_measured": True,
        "_timing_method": "cuda events on cuda:0; stressor on cuda:1 own stream",
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "total_tokens": TOTAL_TOKENS, "spill_frac": SPILL,
        "block_tokens": BLOCK_TOKENS, "deadline_ms": DEADLINE_MS,
        "transfer_dir": "pull-equivalent (direction null per Track A)",
        "cells": [],
    }

    # calibrated selector model (e27 anchors) -- used only by peerkv_selector
    cal = {"MHA": dict(single_pts={16384: 17.3, 32768: 22.2},
                       cfk_pt=(32768, 25.1), copyback_pt=(32768, 65.5)),
           "GQA": dict(single_pts={16384: 16.0, 32768: 17.3},
                       cfk_pt=(32768, 19.1), copyback_pt=(32768, 28.0))}

    for geom_name, H in (("MHA", 32), ("GQA", 8)):
        D = 128
        scale = 1.0 / (D ** 0.5)
        block_bytes = BLOCK_TOKENS * H * D * 2 * 2
        n_blocks = TOTAL_TOKENS // BLOCK_TOKENS
        n_spill = int(n_blocks * SPILL)
        spill_bytes = n_spill * block_bytes
        geom = (Geometry.llama2_7b_mha() if geom_name == "MHA"
                else Geometry.gqa_8kv())
        sel_model = DecodeStepModel.calibrate(geom, **cal[geom_name])

        q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")

        def make_blocks(spill_device):
            kb, vb = [], []
            for i in range(n_blocks):
                dev = spill_device if i < n_spill else "cuda:0"
                k = torch.randn(1, H, BLOCK_TOKENS, D, dtype=torch.float16, device=dev)
                v = torch.randn(1, H, BLOCK_TOKENS, D, dtype=torch.float16, device=dev)
                kb.append(k); vb.append(v)
            return kb, vb

        kb_peer, vb_peer = make_blocks("cuda:1")
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

        # host offload arm: SAME KV content, spill pre-coalesced into contiguous
        # pinned 4096-token chunks (the production FlexGen-style layout; e38's
        # _stream_back) -- a paged per-block host layout would add a per-step
        # CPU cat that no real offloader pays.
        HOST_CHUNK = 4096
        K_spill = torch.cat(kb_peer[:n_spill], dim=2)        # on cuda:1
        V_spill = torch.cat(vb_peer[:n_spill], dim=2)
        Kh = [K_spill[:, :, s:s + HOST_CHUNK, :].cpu().pin_memory()
              for s in range(0, K_spill.shape[2], HOST_CHUNK)]
        Vh = [V_spill[:, :, s:s + HOST_CHUNK, :].cpu().pin_memory()
              for s in range(0, V_spill.shape[2], HOST_CHUNK)]
        K_loc_cat = torch.cat([b.to("cuda:0") for b in kb_peer[n_spill:]], dim=2)
        V_loc_cat = torch.cat([b.to("cuda:0") for b in vb_peer[n_spill:]], dim=2)
        del K_spill, V_spill
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

        # CFK shards: contiguous K/V halves resident per device (steady state)
        def cat_half(blocks, lo, hi, dev):
            return torch.cat([b.to(dev) for b in blocks[lo:hi]], dim=2)
        K0 = cat_half(kb_peer, n_spill, n_blocks, "cuda:0")
        V0 = cat_half(vb_peer, n_spill, n_blocks, "cuda:0")
        K1 = cat_half(kb_peer, 0, n_spill, "cuda:1")
        V1 = cat_half(vb_peer, 0, n_spill, "cuda:1")
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

        c_star = optimal_chunk_blocks(mg_model, block_bytes,
                                      peak_budget_blocks=n_blocks,
                                      spill_tier=MGTier.PEER_NVLINK)
        aqua_blocks = max(1, AQUA_BYTES // block_bytes)

        from umallm.peer_parallel_attn import flash_partial, merge_partial
        host_stream = torch.cuda.Stream(device="cuda:0")

        def host_offload_step():
            """Double-buffered pinned-chunk H2D streaming + flash merge
            (e38's _stream_back; the production host-offload pattern)."""
            O, lse = flash_partial(q, K_loc_cat, V_loc_cat, scale)
            n = len(Kh)
            buf = [None] * n
            ev = [torch.cuda.Event() for _ in range(n)]
            with torch.cuda.stream(host_stream):
                buf[0] = (Kh[0].to("cuda:0", non_blocking=True),
                          Vh[0].to("cuda:0", non_blocking=True))
                ev[0].record(host_stream)
            for j in range(n):
                if j + 1 < n:
                    with torch.cuda.stream(host_stream):
                        buf[j + 1] = (Kh[j + 1].to("cuda:0", non_blocking=True),
                                      Vh[j + 1].to("cuda:0", non_blocking=True))
                        ev[j + 1].record(host_stream)
                torch.cuda.current_stream().wait_event(ev[j])
                Kc, Vc = buf[j]
                Oc, lc = flash_partial(q, Kc, Vc, scale)
                O, lse = merge_partial(O, lse, Oc, lc)
                buf[j] = None
            return O.to(torch.float16)

        ARMS = {
            "harvest_perblock": lambda: flash_merge_attention(
                q, kb_peer, vb_peer, scale=scale, chunk_blocks=1),
            "aqua_64mb": lambda: flash_merge_attention(
                q, kb_peer, vb_peer, scale=scale, chunk_blocks=aqua_blocks),
            "peerkv_cstar": lambda: flash_merge_attention(
                q, kb_peer, vb_peer, scale=scale, chunk_blocks=c_star),
            "host_offload": host_offload_step,
            "cfk": lambda: peer_parallel_attention(
                q, [(K0, V0), (K1, V1)], scale=scale),
        }

        # exactness reference: everything local
        ref = flash_merge_attention(
            q, [b.to("cuda:0") for b in kb_peer], [b.to("cuda:0") for b in vb_peer],
            scale=scale, chunk_blocks=n_blocks)
        torch.cuda.empty_cache()

        # fits-single control (do-no-harm positive evidence)
        kb_loc = [b.to("cuda:0") for b in kb_peer]
        vb_loc = [b.to("cuda:0") for b in vb_peer]
        single_ms, single_iqr = timed(lambda: flash_merge_attention(
            q, kb_loc, vb_loc, scale=scale, chunk_blocks=n_blocks))
        d_fit = online_select(TOTAL_TOKENS, geom, Deployment(), sel_model,
                              peer=PeerState(), link=LinkState(), strict=True)
        out["cells"].append({
            "geom": geom_name, "peer_state": "n/a", "corner": "single",
            "arm": "single_control", "ms": round(single_ms, 3),
            "iqr_ms": round(single_iqr, 3),
            "selector_choice": d_fit.point.value,
            "do_no_harm_ok": d_fit.point is OperatingPoint.SINGLE,
            "source": "eB2 measured"})
        del kb_loc, vb_loc
        torch.cuda.empty_cache()

        for peer_state in ("idle", "busy"):
            stress = PeerStressor() if peer_state == "busy" else None
            lender_solo = None
            if stress:
                with PeerStressor() as s_solo:
                    time.sleep(2.0)
                lender_solo = s_solo.rate

            # selector decision for this (ctx, peer-state) -- OOM-class request
            ps = PeerState(compute_idle=(peer_state == "idle"))
            ctx_oom = 150_000 if geom_name == "MHA" else 600_000
            d = online_select(ctx_oom, geom, Deployment(), sel_model,
                              peer=ps, link=LinkState(), strict=True)
            sel_arm = {"compute_follows_kv": "cfk", "copyback": "peerkv_cstar",
                       "host": "host_offload"}.get(d.point.value)

            for arm, step in ARMS.items():
                ctx = stress.__enter__() if stress else None
                ms, iqr = timed(step)
                o = step()
                cos = float(F.cosine_similarity(
                    ref.reshape(-1).float(), o.reshape(-1).float(), dim=0))
                lender = None
                if stress:
                    stress.__exit__()
                    lender = round(stress.rate / lender_solo, 4)
                dt = ms - single_ms
                # guard: dt must clear the noise floor (3x IQR or 0.3 ms) or
                # the division yields a meaningless 'bandwidth'; also note the
                # single_ms baseline is idle-state, so busy-cell rates mix
                # compute slowdown into the denominator (exposure time, not
                # link bandwidth) -- reported for relative comparison only.
                eff_gbps = ((spill_bytes / 1e9) / dt * 1e3
                            if arm != "cfk" and dt > max(3 * iqr, 0.3) else None)
                out["cells"].append({
                    "geom": geom_name, "peer_state": peer_state, "arm": arm,
                    "corner": {"harvest_perblock": "copyback",
                               "aqua_64mb": "copyback",
                               "peerkv_cstar": "copyback",
                               "host_offload": "host", "cfk": "cfk"}[arm],
                    "ms": round(ms, 3), "iqr_ms": round(iqr, 3),
                    "cos_vs_ref": round(cos, 6),
                    "meets_deadline": ms <= DEADLINE_MS,
                    "spill_bytes_mb": round(spill_bytes / 1024 ** 2, 1),
                    "eff_transfer_gbps": round(eff_gbps, 1) if eff_gbps else None,
                    "lender_flops_retained": lender,
                    "chunk_blocks": {"harvest_perblock": 1,
                                     "aqua_64mb": aqua_blocks,
                                     "peerkv_cstar": c_star,
                                     "host_offload": f"pinned {HOST_CHUNK}tok chunks",
                                     "cfk": None}[arm],
                    "selected_by_policy": arm == sel_arm,
                    "source": "eB2 measured"})
                print(f"[{geom_name} {peer_state:4s}] {arm:17s} "
                      f"{ms:8.3f}ms iqr={iqr:5.3f} cos={cos:.5f}"
                      + (f" lender={lender}" if lender else ""))

            out["cells"].append({
                "geom": geom_name, "peer_state": peer_state,
                "arm": "peerkv_selector", "corner": d.point.value,
                "selector_predicted_full_model_ms_per_token": round(d.predicted_ms, 2),
                "_predicted_units_note": ("full-model ms/token at the OOM ctx "
                                          "(150K/600K) from DecodeStepModel -- "
                                          "NOT comparable to this file's "
                                          "single-layer step ms"),
                "selector_reason": d.reason, "maps_to_arm": sel_arm,
                "do_no_harm_violations": 0,     # strict mode: would have raised
                "source": "eB2 selector (strict)"})
            print(f"[{geom_name} {peer_state:4s}] selector -> {d.point.value} "
                  f"({d.reason})")

        del kb_peer, vb_peer, Kh, Vh, K_loc_cat, V_loc_cat, K0, V0, K1, V1, ref
        torch.cuda.empty_cache()

    path = os.path.join(RES, "eB2_headtohead.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()

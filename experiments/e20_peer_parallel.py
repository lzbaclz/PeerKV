"""e20 -- PeerKV-Parallel: distributed partial attention, partials-only over NVLink.

The improved design. Instead of moving cold KV from the peer GPU back to the
compute GPU every decode step (e17/e18: transfer-bound, GBs over NVLink), we keep
cold KV RESIDENT on the peer GPU and compute its attention *partial* there, moving
only the KB-sized online-softmax statistics (m, l, O) back over NVLink.

Per decode step:
  1. broadcast q (~KB) to cuda:1
  2. in parallel: cuda:0 computes partial over its local KV; cuda:1 computes partial
     over its resident (cold/overflow) KV -- each reads its OWN HBM (~773 GB/s),
     NOT over NVLink
  3. move cuda:1's partial (m1,l1,O1) ~= a few KB back over NVLink
  4. cuda:0 merges the two partials (online-softmax) -> exact attention output

This replaces "move ~GB of KV over NVLink (273 GB/s)" with "compute on the peer
reading its own HBM + move ~KB", and uses the peer GPU's otherwise-idle compute and
its full local HBM bandwidth. With a balanced split the two GPUs' HBM bandwidths add,
so an overflow context can decode at ~single-GPU latency (or faster).

Compares, at a context that overflows one GPU's KV budget:
  * peer-parallel  (ours, resident KV, partials-only)        [SOTA candidate]
  * chunked copy-back (current design, move peer KV -> cuda:0, C blocks/transfer)
  * host-offload   (move host KV -> cuda:0)                   [baseline]
  * all-local      (all KV on cuda:0; reference, OOMs at true scale)
reporting per-step latency, cuda:0 peak, and exactness vs dense attention.

Run on the dual-A100 box (NVLink active):
    python experiments/e20_peer_parallel.py --n-blocks 2048 --local-blocks 1024
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "peer_parallel.json"


def _write(res: dict) -> None:
    res["_experiment"] = "e20_peer_parallel"
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


def _partial(q, K, V, scale):
    """Flash partial over one device's KV -> (O, lse).

    Uses the fused fp16 flash kernel (aten._scaled_dot_product_flash_attention),
    which returns the normalized output O:(1,H,1,D) and the log-sum-exp
    lse:(1,H,1). Falls back to a math implementation if the op is unavailable.
    These (O, lse) partials merge exactly via :func:`_merge_pair` (ring/flash merge).
    """
    import torch
    try:
        r = torch.ops.aten._scaled_dot_product_flash_attention(
            q.contiguous(), K.contiguous(), V.contiguous(),
            0.0, False, False, scale=scale)
        return r[0], r[1]                                   # O:(1,H,1,D), lse:(1,H,1)
    except Exception:
        s = (q.float() * scale) @ K.float().transpose(-1, -2)   # (1,H,1,T)
        lse = torch.logsumexp(s, dim=-1)                        # (1,H,1)
        O = (torch.softmax(s, dim=-1) @ V.float()).to(q.dtype)  # (1,H,1,D)
        return O, lse


def _merge_pair(O0, lse0, O1, lse1):
    """Exact ring/flash merge of two (O, lse) partials -> (O, lse), fp32 O."""
    import torch
    lse = torch.logaddexp(lse0, lse1)                       # (1,H,1)
    w0 = torch.exp(lse0 - lse).unsqueeze(-1)                # (1,H,1,1)
    w1 = torch.exp(lse1 - lse).unsqueeze(-1)
    return O0.float() * w0 + O1.float() * w1, lse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=2048)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--local-blocks", type=int, default=1024,
                    help="blocks on cuda:0; rest resident on cuda:1 (peer)")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunk-blocks", type=int, default=128,
                    help="chunk size for the copy-back baseline")
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    res = {k: getattr(args, k) for k in
           ("n_blocks", "block_tokens", "local_blocks", "heads", "head_dim",
            "chunk_blocks", "trials")}
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # noqa: BLE001
        res.update({"_is_measured": False, "note": f"torch import failed: {e}"})
        _write(res); return

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        res.update({"_is_measured": False,
                    "note": f"need >=2 CUDA GPUs (have {ngpu})"})
        _write(res); print(f"  only {ngpu} GPU(s) -- placeholder."); return

    H, T, D = args.heads, args.block_tokens, args.head_dim
    N, L = args.n_blocks, min(args.local_blocks, args.n_blocks)
    P = N - L                                   # peer/overflow blocks
    scale = 1.0 / (D ** 0.5)
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))
    torch.manual_seed(0)

    def randblk(dev, n):
        return torch.randn(1, H, n * T, D, dtype=torch.float16, device=dev)

    # Resident layout: local KV on cuda:0, cold/overflow KV on cuda:1 (never moved).
    q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")
    K0 = randblk("cuda:0", L); V0 = randblk("cuda:0", L)
    K1 = randblk("cuda:1", P); V1 = randblk("cuda:1", P)
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def sync_both():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    # ---- exactness vs dense attention over the full KV (fp32 reference) ----
    Kfull = torch.cat([K0, K1.to("cuda:0")], dim=2)
    Vfull = torch.cat([V0, V1.to("cuda:0")], dim=2)
    ref = F.scaled_dot_product_attention(q.float(), Kfull.float(), Vfull.float(),
                                         scale=scale)

    def peer_parallel_step():
        q1 = q.to("cuda:1", non_blocking=True)            # ~KB over NVLink (in)
        O1, lse1 = _partial(q1, K1, V1, scale)            # on cuda:1 (peer HBM)
        O0, lse0 = _partial(q, K0, V0, scale)             # on cuda:0 (local HBM)
        # move peer partial (O1, lse1) ~= a few KB over NVLink (out)
        O1c = O1.to("cuda:0", non_blocking=True)
        lse1c = lse1.to("cuda:0", non_blocking=True)
        out, _ = _merge_pair(O0, lse0, O1c, lse1c)
        return out

    out = peer_parallel_step(); sync_both()
    cos = float(torch.nn.functional.cosine_similarity(
        ref.reshape(-1), out.reshape(-1).float(), dim=0))
    max_abs = float((ref - out.float()).abs().max())
    res["exactness_cosine"] = cos
    res["exactness_max_abs_err"] = max_abs
    print(f"  exactness vs dense: cos={cos:.6f}  max_abs={max_abs:.2e}")

    partial_bytes = (H * D) * 2 + (H * 1) * 4  # O fp16 + lse fp32 over NVLink/step
    res["peer_partial_transfer_bytes_per_layer"] = partial_bytes
    res["peer_partial_transfer_bytes_32layer"] = partial_bytes * 32

    # Free the exactness reference tensors so they do NOT pollute the peak
    # measurement (the deployed peer-parallel step holds only q + local KV +
    # the KB partial; Kfull/Vfull/ref are an eval-only artifact).
    del Kfull, Vfull, ref, out
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    # ---- timing helpers ----
    def bench(step_fn, warmup=5):
        for _ in range(warmup):
            step_fn()
        sync_both()
        ts = []
        for _ in range(args.trials):
            sync_both(); t0 = time.perf_counter()
            step_fn(); sync_both()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts) * 1e3

    # peer-parallel (ours)
    torch.cuda.reset_peak_memory_stats(0)
    t_pp = bench(peer_parallel_step)
    pp_peak = torch.cuda.max_memory_allocated(0) / 1024**2

    # chunked copy-back baseline (current design): move peer KV -> cuda:0 in C-block
    # chunks, flash-merge. KV stays resident on cuda:1 between steps.
    C = max(1, args.chunk_blocks)
    Ctok = C * T

    def copyback_step(src_K, src_V):
        # local partial on cuda:0, then stream peer/host KV back in C-block chunks,
        # folding each chunk into the running (O, lse) -> bounded peak (q + one chunk).
        O_acc, lse_acc = _partial(q, K0, V0, scale)
        nt = src_K.shape[2]
        for s in range(0, nt, Ctok):
            Kc = src_K[:, :, s:s+Ctok, :].to("cuda:0", non_blocking=True)
            Vc = src_V[:, :, s:s+Ctok, :].to("cuda:0", non_blocking=True)
            Oc, lc = _partial(q, Kc, Vc, scale)
            O_acc, lse_acc = _merge_pair(O_acc, lse_acc, Oc, lc)
        return O_acc

    def copyback_final():
        return copyback_step(K1, V1)

    torch.cuda.reset_peak_memory_stats(0)
    t_cb = bench(copyback_final)
    cb_peak = torch.cuda.max_memory_allocated(0) / 1024**2

    # host-offload baseline: cold KV on host (pinned), streamed back in chunks
    Kh = K1.to("cpu").pin_memory(); Vh = V1.to("cpu").pin_memory()

    def host_final():
        return copyback_step(Kh, Vh)
    torch.cuda.reset_peak_memory_stats(0)
    t_host = bench(host_final)

    # all-local reference (all KV on cuda:0): only if it fits
    try:
        Kall = torch.cat([K0, K1.to("cuda:0")], dim=2)
        Vall = torch.cat([V0, V1.to("cuda:0")], dim=2)

        def all_local_step():
            return F.scaled_dot_product_attention(q, Kall, Vall, scale=scale)
        torch.cuda.reset_peak_memory_stats(0)
        t_loc = bench(all_local_step)
        loc_peak = torch.cuda.max_memory_allocated(0) / 1024**2
        del Kall, Vall
    except RuntimeError as e:
        t_loc, loc_peak = None, None
        print(f"  all-local OOM/err: {str(e)[:80]}")
    torch.cuda.empty_cache()

    res.update({
        "_is_measured": True, "kind": "peer_parallel_decode_step",
        "device": torch.cuda.get_device_name(0), "n_gpus": ngpu,
        "peer_access_enabled": p2p, "peer_blocks": P,
        "peer_mb": P * (T*D*2*H) * 2 / 1024**2,   # *2 = K and V (was K-only)
        "peer_parallel_ms": t_pp, "peer_parallel_peak_mb": pp_peak,
        "chunked_copyback_ms": t_cb, "chunked_copyback_peak_mb": cb_peak,
        "host_offload_ms": t_host,
        "all_local_ms": t_loc, "all_local_peak_mb": loc_peak,
        "speedup_vs_copyback": (t_cb / t_pp) if t_pp else None,
        "speedup_vs_host": (t_host / t_pp) if t_pp else None,
        "speedup_vs_all_local": (t_loc / t_pp) if (t_loc and t_pp) else None,
        "note": ("peer-parallel keeps cold KV resident on cuda:1 and moves only the "
                 "~KB online-softmax partial over NVLink; baselines move the KV itself. "
                 "speedup_vs_all_local>1 means 2-GPU KV-parallel decode beats single-GPU."),
    })
    _write(res)
    print(f"  peer_parallel={t_pp:.2f}ms  copyback(C={C})={t_cb:.2f}ms  "
          f"host={t_host:.2f}ms  all_local={t_loc}")
    print(f"  speedups: vs_copyback={res['speedup_vs_copyback']}, "
          f"vs_host={res['speedup_vs_host']}, vs_all_local={res['speedup_vs_all_local']}")


if __name__ == "__main__":
    main()

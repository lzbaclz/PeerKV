"""KV-parallel distributed attention: compute attention where the KV lives.

The improved PeerKV design (``PeerKV-Parallel``). The original design spills cold
KV to peer-GPU HBM and **moves it back** over NVLink every decode step -- which our
A100 measurements show is transfer-bound (GBs over NVLink, ~7-11 ms/step even
chunked). Here we instead keep each KV shard RESIDENT on its device (local HBM, or
a peer GPU over NVLink) and compute that shard's attention **partial on its own
device**, moving only the KB-sized online-softmax statistics (the normalized output
``O`` and the log-sum-exp ``lse``) to the query's device, where they are merged.

Per decode step we move ~``H*D*2 + H*4`` bytes per shard (a few KB), never the KV
(GBs). This (a) replaces NVLink KV transfer with peer-local HBM reads, (b) uses the
peer GPU's otherwise-idle compute, and (c) lets the two GPUs' HBM bandwidths add, so
a balanced split decodes an overflow context *faster than a single GPU* while
remaining numerically exact (ring/flash merge).

Measured on dual-A100 NVLink, per-op, **MHA geometry (H=HKV=32)**, 2048 blocks
(2 GB KV), exact (cos 1.000 vs dense). The factor depends on the split (committed
artifacts in ``experiments/results/``):
  - balanced 1024/1024 (``e25_L1024_P1024.json``): 7.25x vs copy-back, 52.3x vs
    host-offload, 1.63x vs single-GPU all-local;
  - peer-heavy 128/1920 (``e25_L128_P1920.json``):  7.74x / 61.7x / 1.03x;
  - mostly-local 1792/256, peer only 128 MB (``peer_parallel.json``): 2.18x vs
    copy-back (the peer shard is tiny, so little is saved).
**These margins are MHA-only.** Under GQA (the modern default, e.g. Llama-3 with
H=32/HKV=8) the lighter KV shrinks the copy-back margin to ~1.4x and a *fitting*
single GPU is 2.7-3.8x faster than KV-parallel (see ``EXPERIMENT_STATUS.md`` /
``e31``). Never quote a margin without its geometry: the single-GPU margin is
largest at a balanced MHA split and ~parity (or a single-GPU win) under GQA; the
full-model (real-weights) end-to-end picture is geometry-dependent and is the
honest test -- see ``experiments/results/real_weights*.json`` /
``A100_PEERKV_REPORT.md``, not this per-op microbenchmark.

The merge is the standard flash/ring-attention reduction and is exact; validated on
CPU in ``tests/test_peer_parallel.py`` (no GPU needed -- this is the math).
"""
from __future__ import annotations

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def flash_partial(q, K, V, scale=None):
    """Attention partial over one KV shard, ON ``K``'s device -> ``(O, lse)``.

    Uses the fused fp16 flash kernel on CUDA (returns normalized output + lse);
    falls back to an exact math implementation off-CUDA. ``O`` is ``(B,H,Lq,D)``,
    ``lse`` is ``(B,H,Lq)``; the two merge exactly via :func:`merge_partial`.
    """
    import torch
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    if K.shape[1] != q.shape[1]:            # GQA: expand kv heads to q heads (exact)
        rep = q.shape[1] // K.shape[1]
        K = K.repeat_interleave(rep, dim=1)
        V = V.repeat_interleave(rep, dim=1)
    if q.is_cuda and K.is_cuda:
        try:
            r = torch.ops.aten._scaled_dot_product_flash_attention(
                q.contiguous(), K.contiguous(), V.contiguous(),
                0.0, False, False, scale=scale)
            return r[0], r[1]                                   # (B,H,Lq,D), (B,H,Lq)
        except Exception:  # pragma: no cover - kernel/dtype unsupported
            pass
    s = (q.float() * scale) @ K.float().transpose(-1, -2)       # (B,H,Lq,Tk)
    lse = torch.logsumexp(s, dim=-1)                            # (B,H,Lq)
    O = (torch.softmax(s, dim=-1) @ V.float()).to(q.dtype)      # (B,H,Lq,D)
    return O, lse


def merge_partial(O0, lse0, O1, lse1):
    """Exact ring/flash merge of two ``(O, lse)`` partials -> ``(O, lse)`` (O fp32)."""
    import torch
    lse = torch.logaddexp(lse0, lse1)
    w0 = torch.exp(lse0 - lse).unsqueeze(-1)
    w1 = torch.exp(lse1 - lse).unsqueeze(-1)
    return O0.float() * w0 + O1.float() * w1, lse


def _causal_partial(q, K, V, scale):
    """Partial over a shard where ``q`` and ``K`` are the SAME positions (the
    diagonal of a ring): query local index p attends to key local indices <= p.
    Returns ``(O, lse)``. Uses the fused causal flash kernel on CUDA; exact masked
    math off-CUDA.
    """
    import torch
    if K.shape[1] != q.shape[1]:            # GQA: expand kv heads to q heads (exact)
        rep = q.shape[1] // K.shape[1]
        K = K.repeat_interleave(rep, dim=1)
        V = V.repeat_interleave(rep, dim=1)
    if q.is_cuda and K.is_cuda:
        try:
            r = torch.ops.aten._scaled_dot_product_flash_attention(
                q.contiguous(), K.contiguous(), V.contiguous(),
                0.0, True, False, scale=scale)             # is_causal=True
            return r[0], r[1]
        except Exception:  # pragma: no cover
            pass
    s = (q.float() * scale) @ K.float().transpose(-1, -2)   # (B,H,Sq,Sk)
    Sq, Sk = s.shape[-2], s.shape[-1]
    mask = torch.triu(torch.ones(Sq, Sk, device=s.device, dtype=torch.bool),
                      diagonal=1)                           # disallow key>query
    s = s.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    O = (torch.softmax(s, dim=-1) @ V.float()).to(q.dtype)
    return O, lse


def ring_prefill_attention(shards, scale=None):
    """Exact 2+-GPU causal *prefill* (ring / context parallelism), KV kept sharded.

    ``shards``: list of ``(Q, K, V)`` in sequence order, each shard's Q/K/V for a
    contiguous position range resident on its own device. Causal attention: query
    shard ``i`` attends to all *earlier* key shards ``0..i-1`` (full, no mask) plus
    its own shard ``i`` (causal). For 2 GPUs and a causal mask, only the
    earlier->later direction needs an exchange (later keys are in the future), so
    shard 0 is fully local and shard 1 pulls shard 0's KV (streamed, bounded peak).

    Returns a list of outputs ``O_i`` on each shard's device; concatenated in order
    they equal dense causal attention over the full sequence. Exact (flash/ring
    merge); validated on CPU in tests/test_peer_parallel.py.
    """
    import torch
    if scale is None:
        scale = 1.0 / (shards[0][0].shape[-1] ** 0.5)
    outs = []
    for i, (Qi, Ki, Vi) in enumerate(shards):
        dev = Qi.device
        O_acc, lse_acc = _causal_partial(Qi, Ki, Vi, scale)    # diagonal (causal)
        for j in range(i):                                     # earlier key shards
            _, Kj, Vj = shards[j]
            Kc = Kj.to(dev, non_blocking=True)                 # ring exchange (KV)
            Vc = Vj.to(dev, non_blocking=True)
            O, lse = flash_partial(Qi, Kc, Vc, scale)          # full (no mask)
            O_acc, lse_acc = merge_partial(O_acc, lse_acc, O, lse)
        outs.append(O_acc.to(Qi.dtype))
    return outs


def peer_parallel_attention(q, kv_shards, scale=None):
    """Exact attention over KV shards that live on (possibly) different devices.

    Args:
        q: ``(B,H,Lq,D)`` on the compute device.
        kv_shards: list of ``(K, V)``, each ``(B,H,Tk,D)`` resident on any device
            (e.g. local HBM and a peer GPU over NVLink). The KV is **not moved**.
        scale: softmax scale; defaults to ``1/sqrt(D)``.

    Returns ``(B,H,Lq,D)`` on ``q``'s device. Each shard's partial is computed on
    its own device; only the ``(O, lse)`` partials (a few KB) cross the link, then
    are merged. Exact vs dense attention over the concatenated KV.
    """
    import torch
    if not kv_shards:
        raise ValueError("kv_shards must be non-empty")
    dev = q.device
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    O_acc = lse_acc = None
    for K, V in kv_shards:
        qd = q.to(K.device, non_blocking=True) if K.device != dev else q
        O, lse = flash_partial(qd, K, V, scale)
        if O.device != dev:                                    # move ~KB, not the KV
            O = O.to(dev, non_blocking=True)
            lse = lse.to(dev, non_blocking=True)
        if O_acc is None:
            O_acc, lse_acc = O.float(), lse
        else:
            O_acc, lse_acc = merge_partial(O_acc, lse_acc, O, lse)
    return O_acc.to(q.dtype)

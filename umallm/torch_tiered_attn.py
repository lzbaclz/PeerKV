"""Block-streaming flash attention for multi-GPU KV tiering (torch, device-agnostic).

The merge math behind e18. Attention over a KV cache given as a LIST OF BLOCKS
that may live on different devices (local HBM / peer GPU over NVLink / host).
We stream block-by-block on the query's device with an online-softmax (flash)
merge, so the peak holds only the query plus ONE block -- never the full KV.
That bounded peak is what lets a context overflowing one GPU run on two: cold
blocks stay on the peer GPU and are pulled a block at a time over NVLink.

Exact vs dense attention (flash merge); validated on CPU in
tests/test_torch_tiered_attn.py (no GPU needed -- this is the math). The
multi-GPU placement + peak/latency measurement is driven by
experiments/e18_p2p_flash.py (needs the dual-A100 box).

NOTE on "P2P direct read": moving each block to the compute device with
``.to()`` uses NVLink for a peer-GPU block (a one-block copy, so peak stays
bounded). A true *zero-copy* read of peer HBM from the attention kernel needs a
custom CUDA kernel and is left as future work; this module already delivers the
bounded-peak enablement via one-block-at-a-time streaming.
"""
from __future__ import annotations

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def flash_merge_attention(q, k_blocks, v_blocks, scale=None, chunk_blocks=1):
    """Online-softmax attention over KV blocks that may be on different devices.

    Args:
        q: (B, H, Lq, D) on the compute device.
        k_blocks, v_blocks: lists of (B, H, Tb, D) tensors on any device(s).
        scale: softmax scale; defaults to 1/sqrt(D).
        chunk_blocks: coalesce up to this many *consecutive same-device* blocks
            into ONE cross-device transfer before merging. ``chunk_blocks=1`` is
            the original block-at-a-time streaming. Larger values make the fetch
            bandwidth-bound instead of launch-bound, so the peer-GPU/NVLink link's
            ~10x bandwidth over host/PCIe is actually realized end-to-end (per-
            block fetch hides it: hundreds of ~0.5MB copies/step are latency-bound
            and NVLink ~= host). Peak still holds only q + ONE chunk (chunk_blocks
            blocks), never the full KV -- so a context overflowing one GPU still
            runs on two; chunk_blocks just trades a larger bounded peak for a
            bandwidth-bound transfer (see experiments/results/p2p_flash_chunked).

    Returns (B, H, Lq, D) on q's device. Coalescing concatenates the chunk on its
    *source* device (a cheap local copy) and issues a single copy to q's device.
    Exact (flash merge), independent of chunk_blocks; decode (Lq==1) needs no mask.

    NOTE: the default chunk_blocks=1 is the *exact-but-launch-bound* path (it
    loses to host for a peer-GPU spill tier). Production callers should pass the
    cost-model-selected C* (``umallm.multigpu.optimal_chunk_blocks``) so peer
    fetches are bandwidth-bound; see experiments/results/p2p_flash_chunked.json.
    """
    import torch
    dev = q.device
    B, H, Lq, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    qs = q * scale
    m = torch.full((B, H, Lq, 1), float("-inf"), device=dev, dtype=torch.float32)
    l = torch.zeros((B, H, Lq, 1), device=dev, dtype=torch.float32)
    o = torch.zeros((B, H, Lq, D), device=dev, dtype=torch.float32)
    n = len(k_blocks)
    c = max(1, int(chunk_blocks))
    i = 0
    while i < n:
        # gather a run of up to c consecutive blocks that live on the SAME device
        src = k_blocks[i].device
        j = i
        while j < n and (j - i) < c and k_blocks[j].device == src:
            j += 1
        if j - i == 1:
            Kb = k_blocks[i].to(dev, non_blocking=True)
            Vb = v_blocks[i].to(dev, non_blocking=True)
        else:                                              # coalesce: cat on src, 1 copy
            Kb = torch.cat(list(k_blocks[i:j]), dim=2).to(dev, non_blocking=True)
            Vb = torch.cat(list(v_blocks[i:j]), dim=2).to(dev, non_blocking=True)
        s = (qs @ Kb.transpose(-1, -2)).float()            # (B,H,Lq,Tc)
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        corr = torch.exp(m - m_new)
        p = torch.exp(s - m_new)                           # (B,H,Lq,Tc)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        o = o * corr + p.to(Vb.dtype) @ Vb                 # (B,H,Lq,D)
        m = m_new
        del Kb, Vb, s, p
        i = j
    return (o / l).to(q.dtype)

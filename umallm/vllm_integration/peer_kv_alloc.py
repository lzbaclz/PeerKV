"""M2: place a fraction of vLLM's paged KV blocks on the peer GPU.

vLLM v1 allocates the whole per-layer KV cache as one ``torch.zeros(kv_cache_shape)``
on ``self.device`` inside ``GPUModelRunner.initialize_kv_cache``
(gpu_model_runner.py:1722, inside :1689). A single torch tensor lives on ONE device,
so *fractional* tiering needs TWO tensors per layer:

  * local(+scratch) on cuda:0  -- the blocks the kernel reads in place + a scratch
    region for staged-in peer blocks (see peerkv_staging.PeerKVLayout)
  * peer on cuda:1             -- the cold block fraction, reachable over NVLink

This patch wraps ``initialize_kv_cache``: it lets vLLM compute ``num_gpu_blocks``
(the profiled ceiling), then re-lays-out each layer as (local+scratch, peer),
publishes the layout via PEERKV_* env (read by peerkv_attn), and hands each layer's
peer tensor to its PeerKVFlashAttentionImpl via ``set_peer_kv``.

``peer_fraction`` is the share of blocks moved to the peer (0.0 = vanilla vLLM).

STATUS: allocation math + handoff concrete; hooks CONFIRMED against installed
vLLM 0.8.5 source -- runner.kv_caches:list[Tensor] (gpu_model_runner.py:162),
Attention.impl (layer.py:134), kv layout [2,num_blocks,bs,H,D]. Only the runtime
ordering (relayout must run after kv_caches is populated, before warmup/capture)
is verified on the dedicated box. Import-safe without vLLM.
"""
from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_TARGET = "vllm.v1.worker.gpu_model_runner:GPUModelRunner.initialize_kv_cache"


def _resolve(spec: str):
    import importlib
    modpath, _, qual = spec.partition(":")
    owner = importlib.import_module(modpath)
    parts = qual.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return owner, parts[-1]


def patch_peer_kv_allocation(peer_fraction: float = 0.3,
                             scratch_blocks: int = 256,
                             peer_device: str = "cuda:1",
                             target: str = _DEFAULT_TARGET) -> bool:
    """Wrap initialize_kv_cache to put ``peer_fraction`` of blocks on ``peer_device``.
    Must run before the engine builds its KV cache. Returns True if patched."""
    if not (0.0 < peer_fraction < 1.0):
        logger.info("PeerKV: peer_fraction=%.2f -> vanilla vLLM (no tiering).",
                    peer_fraction)
        return False
    try:
        import torch  # noqa: F401
        owner, name = _resolve(target)
        orig = getattr(owner, name)
    except Exception as e:  # noqa: BLE001
        logger.info("PeerKV: cannot patch %s (%s); no-op.", target, e)
        return False
    if getattr(orig, "_peerkv_wrapped", False):
        return True

    @functools.wraps(orig)
    def wrapper(self, *args, __orig=orig, **kwargs):
        import torch
        ret = __orig(self, *args, **kwargs)   # vLLM does its normal single-device alloc
        try:
            _relayout_to_peer(self, peer_fraction, scratch_blocks, peer_device)
        except Exception as e:  # noqa: BLE001 -- never break the engine on our account
            logger.error("PeerKV: relayout failed (%s); falling back to vanilla "
                         "single-GPU KV. Tiering disabled this run.", e)
        return ret

    wrapper._peerkv_wrapped = True
    setattr(owner, name, wrapper)
    logger.info("PeerKV: initialize_kv_cache wrapped (peer_fraction=%.2f -> %s).",
                peer_fraction, peer_device)
    return True


def _relayout_to_peer(runner, peer_fraction, scratch_blocks, peer_device):
    """Split each layer's KV tensor into local(+scratch)/peer and wire the backend.

    # H100: the three lines marked below are finalized against the live tree:
    #   (1) how to enumerate per-layer (kv_cache tensor, attn impl) pairs,
    #   (2) the kv_cache tensor handle on the runner,
    #   (3) PeerKVFlashAttentionImpl.set_peer_kv handoff.
    """
    import torch
    from .peerkv_staging import PeerKVLayout, alloc_split_kv

    # (2) H100: runner.kv_caches is the list of per-layer (2, num_blocks, bs, H, D)
    kv_caches = getattr(runner, "kv_caches", None)
    if not kv_caches:
        raise RuntimeError("runner.kv_caches not found (H100: confirm attribute)")

    # (1) H100: enumerate the attention impls in layer order to call set_peer_kv
    impls = _enumerate_attn_impls(runner)

    local_dev = str(kv_caches[0].device)
    n_published = False
    for li, kv in enumerate(kv_caches):
        # kv: (2, num_blocks, block_size, H, D) on cuda:0
        two, num_blocks, bs, H, D = kv.shape
        c_peer = int(num_blocks * peer_fraction)
        c_local = num_blocks - c_peer
        lay = PeerKVLayout(c_local=c_local, c_scratch=scratch_blocks, c_peer=c_peer,
                           block_size=bs, num_kv_heads=H, head_size=D,
                           local_device=local_dev, peer_device=peer_device)
        new_local, peer_kv = alloc_split_kv(lay, dtype=kv.dtype)
        # move the cold fraction onto the peer; keep the hot fraction local
        new_local[:, :c_local].copy_(kv[:, :c_local])
        peer_kv.copy_(kv[:, c_local:c_local + c_peer].to(peer_device))
        kv_caches[li] = new_local           # replace the engine's handle
        if not n_published:
            os.environ["PEERKV_C_LOCAL"] = str(c_local)
            os.environ["PEERKV_C_SCRATCH"] = str(scratch_blocks)
            os.environ["PEERKV_C_PEER"] = str(c_peer)
            os.environ["PEERKV_BLOCK"] = str(bs)
            os.environ["PEERKV_LOCAL_DEV"] = local_dev
            os.environ["PEERKV_PEER_DEV"] = peer_device
            n_published = True
        if li < len(impls) and hasattr(impls[li], "set_peer_kv"):
            impls[li].set_peer_kv(peer_kv)   # (3) H100: handoff
        del kv
    torch.cuda.empty_cache()
    logger.info("PeerKV: relaid %d layers; %.0f%% of blocks on %s.",
                len(kv_caches), peer_fraction * 100, peer_device)


def _enumerate_attn_impls(runner):
    """Return per-layer PeerKVFlashAttentionImpl instances in layer order.
    # H100: vLLM keeps Attention modules in the model; walk modules and collect
    # those whose .impl is a PeerKVFlashAttentionImpl (set by peerkv_register)."""
    impls = []
    model = getattr(runner, "model", None)
    if model is None:
        return impls
    for m in model.modules():
        impl = getattr(m, "impl", None)
        if impl is not None and impl.__class__.__name__ == "PeerKVFlashAttentionImpl":
            impls.append(impl)
    return impls

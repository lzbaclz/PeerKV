"""JIT loader + Python wrapper for the multi-device CFK attention CUDA op.

The op lives in ``csrc/peer_fused_attn_ext.cu`` (shares kernels with the standalone
bench via ``csrc/peer_attn_kernels.cuh``). We build it on first use with
``torch.utils.cpp_extension.load`` so no ahead-of-time compile / wheel is required,
and enable peer access between the two devices.

Public API:
  peer_fused_attn(q, K_local, V_local, K_peer, V_peer, scale=None, splits=16) -> O
  reference_attn(q, K_full, V_full, scale=None) -> O      # single-GPU GQA reference

Shapes (fp16):
  q        [H, D]        on the local (compute) GPU
  K/V_local[HKV, Tl, D]  on the local GPU
  K/V_peer [HKV, Tp, D]  on the peer GPU (pass empty tensors / Tp=0 to disable peer)
  returns  [H, D]        on the local GPU
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import torch

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "csrc" / "peer_fused_attn_ext.cu"
_ext = None  # cached compiled module


def load_fused_attn(verbose: bool = False):
    """Build (once) and return the compiled CUDA extension module."""
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load
    assert _SRC.exists(), f"missing CUDA source: {_SRC}"
    _ext = load(
        name="peer_fused_attn_ext",
        sources=[str(_SRC)],
        extra_include_paths=[str(_REPO / "csrc")],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )
    return _ext


def _enable_p2p(dev0: int, dev1: int) -> None:
    """Best-effort enable bidirectional P2P between the two devices."""
    if dev0 == dev1:
        return
    for a, b in ((dev0, dev1), (dev1, dev0)):
        try:
            torch.cuda.set_device(a)
            torch.cuda.synchronize(a)
            # torch enables P2P lazily on cross-device copies; this is a no-op guard.
        except Exception:
            pass


def peer_fused_attn(q: torch.Tensor,
                    K_local: torch.Tensor, V_local: torch.Tensor,
                    K_peer: Optional[torch.Tensor] = None,
                    V_peer: Optional[torch.Tensor] = None,
                    scale: Optional[float] = None,
                    splits: int = 16) -> torch.Tensor:
    """Multi-device CFK attention. If K_peer/V_peer are None/empty, runs single-GPU."""
    ext = load_fused_attn()
    H, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if K_peer is None or K_peer.numel() == 0:
        empty = q.new_empty((K_local.size(0), 0, D))
        K_peer = K_peer if (K_peer is not None and K_peer.numel()) else empty
        V_peer = V_peer if (V_peer is not None and V_peer.numel()) else empty
    else:
        _enable_p2p(q.device.index, K_peer.device.index)
    return ext.peer_fused_attn(q.contiguous(), K_local.contiguous(), V_local.contiguous(),
                               K_peer.contiguous(), V_peer.contiguous(), float(scale), int(splits))


def reference_attn(q: torch.Tensor, K_full: torch.Tensor, V_full: torch.Tensor,
                   scale: Optional[float] = None) -> torch.Tensor:
    """Single-GPU GQA attention reference (fp32 math) for numerics checks.
    q [H,D]; K/V_full [HKV, T, D]. Returns [H, D] fp16."""
    H, D = q.shape
    HKV, T, _ = K_full.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    groups = H // HKV
    qf = q.float()
    out = torch.empty((H, D), dtype=torch.float32, device=q.device)
    for h in range(H):
        kvh = h // groups
        s = (qf[h] @ K_full[kvh].float().T) * scale      # [T]
        w = torch.softmax(s, dim=-1)                       # [T]
        out[h] = w @ V_full[kvh].float()                   # [D]
    return out.half()

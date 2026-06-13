"""Numerics + smoke test for the multi-device CFK attention torch op.

Builds umallm.peerkv.fused_attn (JIT) and checks peer_fused_attn (KV sharded across
cuda:0/cuda:1) matches a single-GPU full-KV reference. Run on a 2-GPU NVLink box:
  python experiments/test_fused_attn_op.py
"""
from __future__ import annotations
import sys
import torch

from umallm.peerkv.fused_attn import peer_fused_attn, reference_attn


def cosine(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    assert torch.cuda.device_count() >= 2, "need 2 GPUs"
    torch.manual_seed(0)
    H, HKV, D = 32, 8, 128
    ok = True
    for T in (1024, 4096, 16384):
        Tl = T // 2; Tp = T - Tl
        q = (torch.randn(H, D, device="cuda:0") * 0.1).half()
        Kf = (torch.randn(HKV, T, D, device="cuda:0") * 0.1).half()
        Vf = (torch.randn(HKV, T, D, device="cuda:0") * 0.1).half()
        # shard per kv-head along the token axis
        K0 = Kf[:, :Tl, :].contiguous(); V0 = Vf[:, :Tl, :].contiguous()
        K1 = Kf[:, Tl:, :].contiguous().to("cuda:1"); V1 = Vf[:, Tl:, :].contiguous().to("cuda:1")

        O = peer_fused_attn(q, K0, V0, K1, V1)          # multi-device CFK
        ref = reference_attn(q, Kf, Vf)                  # single-GPU full KV
        c = cosine(O, ref)
        # also test the Tp==0 single-GPU fast path
        O_single = peer_fused_attn(q, Kf, Vf, None, None)
        c_single = cosine(O_single, ref)
        print(f"T={T:6d}  CFK cos={c:.6f}  single-path cos={c_single:.6f}  "
              f"O.device={O.device}")
        ok = ok and c > 0.999 and c_single > 0.999
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""nsys profiling target: holder decode on GPU1 overlapped with a push handoff
GPU1->GPU0 over NVLink. Used to confirm temporal overlap (decode kernels and the
peer memcpy execute concurrently) for the mechanism evidence (reviewer Q1/Q2).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
B, S = 1, 32768
dt = torch.float16
torch.cuda.set_device(1)
W = {k: torch.randn(*s, dtype=dt, device="cuda:1") * 0.02 for k, s in {
    "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
    "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
Kc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
Vc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
x = torch.randn(B, 1, D, dtype=dt, device="cuda:1") * 0.02
total = 512 * 1024 * 1024 // 2
src_h = torch.randn(total, dtype=dt, device="cuda:1")
dst_c = torch.empty(total, dtype=dt, device="cuda:0")
dec = torch.cuda.Stream(device=1)
cp = torch.cuda.Stream(device=1)


def decode_step():
    q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
    k = (x @ W["k"]).view(B, 1, HKV, HD).transpose(1, 2)
    v = (x @ W["v"]).view(B, 1, HKV, HD).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
    return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]


for _ in range(10):
    with torch.cuda.stream(dec):
        decode_step()
torch.cuda.synchronize(1)
torch.cuda.cudart().cudaProfilerStart()
for _ in range(8):
    with torch.cuda.stream(cp):
        dst_c.copy_(src_h, non_blocking=True)
for _ in range(60):
    with torch.cuda.stream(dec):
        decode_step()
    if cp.query():
        with torch.cuda.stream(cp):
            dst_c.copy_(src_h, non_blocking=True)
torch.cuda.synchronize(0)
torch.cuda.synchronize(1)
torch.cuda.cudart().cudaProfilerStop()
print("prof_target done")

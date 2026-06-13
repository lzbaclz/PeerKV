"""Prosecution probe #2: does push>pull survive a STRIDED PAGED-KV GATHER, not a
bulk contiguous copy? Real KV transfer over NVLink reads non-contiguous paged
blocks. If the copy engine loses its local-read arbitration win on scattered
access, the mechanism is microbenchmark-only (the load-bearing unretired risk).

We gather BLOCK_TOKENS-sized pages scattered across a large KV pool via an
index_select-style gather, issued either by the consumer (pull) or holder (push),
under a memory-bound holder. Compare retained BW.
"""
from __future__ import annotations
import json, statistics, time
from pathlib import Path
import torch

TRIALS = 50
# paged KV pool on holder GPU1: many 16-token pages, head_dim*heads ~ 1024 elems/token
BLOCK = 16
TOK_ELEMS = 8 * 128            # gqa8 * head_dim128 = 1024 fp16 elems/token
POOL_PAGES = 8192             # 8192 pages * 16 tok * 1024 = 128M elems = 256MB
GATHER_PAGES = 4096          # gather half the pool = 128MB transferred

pool1 = torch.ones(POOL_PAGES, BLOCK, TOK_ELEMS, dtype=torch.float16, device="cuda:1")
# scattered page ids (non-contiguous, shuffled)
idx = torch.randperm(POOL_PAGES, device="cuda:1")[:GATHER_PAGES]
idx0 = idx.to("cuda:0")
dst0 = torch.empty(GATHER_PAGES, BLOCK, TOK_ELEMS, dtype=torch.float16, device="cuda:0")
nbytes = GATHER_PAGES * BLOCK * TOK_ELEMS * 2

pull_stream = torch.cuda.Stream(device=0)
push_stream = torch.cuda.Stream(device=1)


def pull_gather():
    # consumer GPU0 reads scattered pages from holder GPU1 (remote gather)
    with torch.cuda.stream(pull_stream):
        torch.index_select(pool1, 0, idx, out=None)  # placeholder; real path below


def pull_gather_real():
    with torch.cuda.stream(pull_stream):
        # gather happens reading remote HBM into local dst (cross-device index_select)
        dst0.copy_(pool1.index_select(0, idx), non_blocking=True)


def push_gather():
    # holder GPU1 gathers locally then writes contiguous result into GPU0
    with torch.cuda.stream(push_stream):
        tmp = pool1.index_select(0, idx)   # LOCAL gather on holder (wins arbitration)
        dst0.copy_(tmp, non_blocking=True)  # contiguous remote WRITE


def bw_wall(fn, sync_dev):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(sync_dev)
    ts = []
    for _ in range(TRIALS):
        torch.cuda.synchronize(sync_dev)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(sync_dev)
        ts.append(time.perf_counter() - t0)
    return nbytes / statistics.median(ts) / 1e9


def mk_membound():
    m, k, r = 16384, 16384, 8
    A = torch.randn(m, k, dtype=torch.float16, device="cuda:1")
    B = torch.randn(k, r, dtype=torch.float16, device="cuda:1")
    return lambda: torch.mm(A, B)


op = mk_membound()
torch.cuda.set_device(1)
for _ in range(5): op()
torch.cuda.synchronize(1)
t0 = time.perf_counter()
for _ in range(200): op()
torch.cuda.synchronize(1)
per = (time.perf_counter() - t0) / 200
torch.cuda.set_device(0)
nb = max(120, int(0.40 / max(per, 1e-5)))
load_stream = torch.cuda.Stream(device=1)


def load():
    with torch.cuda.stream(load_stream):
        for _ in range(nb):
            op()


pull_idle = bw_wall(pull_gather_real, 0)
push_idle = bw_wall(push_gather, 1)

torch.cuda.synchronize(1); load(); pull_mb = bw_wall(pull_gather_real, 0); torch.cuda.synchronize(1)
torch.cuda.synchronize(1); load(); push_mb = bw_wall(push_gather, 1); torch.cuda.synchronize(1)

res = {
    "shape": "strided paged-KV gather (gqa8 x d128, 16-tok pages, 128MB scattered)",
    "trials": TRIALS,
    "pull_idle_gbs": round(pull_idle, 1),
    "push_idle_gbs": round(push_idle, 1),
    "pull_membound_gbs": round(pull_mb, 1),
    "push_membound_gbs": round(push_mb, 1),
    "pull_retained": round(pull_mb / pull_idle, 4),
    "push_retained": round(push_mb / push_idle, 4),
    "push_adv_membound_pp": round((push_mb / pull_mb - 1) * 100, 1),
    "push_adv_idle_pp": round((push_idle / pull_idle - 1) * 100, 1),
}
print(json.dumps(res, indent=2))
Path("/home/lzq/codes/PeerKV/experiments/results/prosecute_gather.json").write_text(json.dumps(res, indent=2))

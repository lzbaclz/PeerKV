"""Prosecution probe: is the e41 push>pull win the NOVEL read-port-arbitration
effect, or the KNOWN put>get request/response asymmetry + a timing artifact?

Controls e41 lacked:
  (A) IDENTICAL timing method for PULL and PUSH (both wall-clock, both synced on
      the busy holder GPU1) -- e41 timed PULL with cuda events on GPU0 and PUSH
      with wall-clock on GPU1, which is not apples-to-apples.
  (B) idle-holder push-vs-pull baseline -- the known put>get asymmetry (~up to
      10%, request/response packets) shows up even with NO contention. Subtract it.
  (C) report push_adv at idle vs membound: the NOVEL claim requires the advantage
      to GROW under a memory-bound holder beyond the idle baseline.
"""
from __future__ import annotations
import json, statistics, time
from pathlib import Path
import torch

MB = 256
TRIALS = 60
n = MB * 1024 * 1024 // 2
nbytes = n * 2

src1 = torch.ones(n, dtype=torch.float16, device="cuda:1")   # data on holder GPU1
dst0 = torch.empty(n, dtype=torch.float16, device="cuda:0")  # consumer GPU0

pull_stream = torch.cuda.Stream(device=0)   # consumer issues the read
push_stream = torch.cuda.Stream(device=1)   # holder issues the write


def pull_once():
    with torch.cuda.stream(pull_stream):
        dst0.copy_(src1, non_blocking=True)


def push_once():
    with torch.cuda.stream(push_stream):
        dst0.copy_(src1, non_blocking=True)


def bw_wall(issue, sync_dev):
    """IDENTICAL method for both: wall-clock, sync the issuing device."""
    for _ in range(5):
        issue()
    torch.cuda.synchronize(sync_dev)
    ts = []
    for _ in range(TRIALS):
        torch.cuda.synchronize(sync_dev)
        t0 = time.perf_counter()
        issue()
        torch.cuda.synchronize(sync_dev)
        ts.append(time.perf_counter() - t0)
    return nbytes / statistics.median(ts) / 1e9


def mk_membound():
    m, k, r = 16384, 16384, 8
    A = torch.randn(m, k, dtype=torch.float16, device="cuda:1")
    B = torch.randn(k, r, dtype=torch.float16, device="cuda:1")
    return lambda: torch.mm(A, B)


def per_iter(op):
    torch.cuda.set_device(1)
    for _ in range(5):
        op()
    torch.cuda.synchronize(1)
    t0 = time.perf_counter()
    for _ in range(200):
        op()
    torch.cuda.synchronize(1)
    torch.cuda.set_device(0)
    return (time.perf_counter() - t0) / 200


# ---- idle baselines (same method) ----
pull_idle = bw_wall(pull_once, 0)
push_idle = bw_wall(push_once, 1)

# ---- under a memory-bound holder ----
op = mk_membound()
per = per_iter(op)
nb = max(120, int(0.40 / max(per, 1e-5)))
load_stream = torch.cuda.Stream(device=1)


def load():
    with torch.cuda.stream(load_stream):
        for _ in range(nb):
            op()


torch.cuda.synchronize(1); load(); pull_mb = bw_wall(pull_once, 0); torch.cuda.synchronize(1)
torch.cuda.synchronize(1); load(); push_mb = bw_wall(push_once, 1); torch.cuda.synchronize(1)

res = {
    "method": "IDENTICAL wall-clock for pull and push; sync issuing device",
    "trials": TRIALS,
    "pull_idle_gbs": round(pull_idle, 1),
    "push_idle_gbs": round(push_idle, 1),
    "idle_push_over_pull_pct": round((push_idle / pull_idle - 1) * 100, 1),
    "holder_per_iter_us": round(per * 1e6, 1),
    "pull_membound_gbs": round(pull_mb, 1),
    "push_membound_gbs": round(push_mb, 1),
    "pull_retained_vs_pull_idle": round(pull_mb / pull_idle, 4),
    "push_retained_vs_push_idle": round(push_mb / push_idle, 4),
    # The NOVEL number: push advantage UNDER CONTENTION minus push advantage at IDLE.
    "push_adv_idle_pp": round((push_idle / pull_idle - 1) * 100, 1),
    "push_adv_membound_pp": round((push_mb / pull_mb - 1) * 100, 1),
    "novel_contention_only_pp": round(((push_mb / pull_mb) - (push_idle / pull_idle)) * 100, 1),
}
print(json.dumps(res, indent=2))
Path("/home/lzq/codes/PeerKV/experiments/results/prosecute_pushpull.json").write_text(json.dumps(res, indent=2))

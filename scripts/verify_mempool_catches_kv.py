#!/usr/bin/env python3
"""Confirm the managed KV allocator actually catches allocations.

Runs on ANY CUDA box (A100/H100 -- not just GH200), so you de-risk the single
most likely Route B failure (the MemPool patch not biting vLLM's KV-allocation
site) BEFORE paying for GH200 time.

Checks, in order:
  1. direct  -- allocate a tensor inside ``uma_backend.kv_alloc_scope()`` and
     confirm the managed pool's ``live_bytes`` grew by ~the tensor size. Proves
     the cudaMallocManaged-backed pluggable allocator is wired to torch.
  2. vllm    -- (optional, ``--model <hf-id>``) ``patch_vllm_kv_allocation()``,
     build a small LLM, confirm the pool grew by ~the KV cache size. Proves the
     patch bites the *real* KV allocation site across the installed vLLM
     version -- the thing most likely to drift.

This validates WIRING, not performance: on a non-GH200 box managed pages
migrate over PCIe (not coherent C2C), so latency means nothing here. Numbers
come from a GH200. Exit: 0 all checks pass, 1 a check failed, 2 unavailable.
"""
from __future__ import annotations

import argparse
import sys

from umallm.uma_alloc import native_available


def _pool_live_bytes():
    import umallm._uma_native as native  # built extension
    return int(native.pool_stats()["live_bytes"])


def check_direct(alloc_mib: int) -> dict:
    import torch
    from umallm.vllm_integration.uma_backend import kv_alloc_scope, uma_mem_pool

    if uma_mem_pool() is None:
        return {"name": "direct_scope", "pass": False,
                "reason": "uma_mem_pool() is None (native allocator not live)"}
    n = (alloc_mib * 1024 * 1024) // 2  # fp16 elements
    before = _pool_live_bytes()
    with kv_alloc_scope():
        t = torch.zeros(n, dtype=torch.float16, device="cuda")
    grew = _pool_live_bytes() - before
    want = t.numel() * t.element_size()
    ok = grew >= 0.9 * want
    del t
    return {"name": "direct_scope", "pass": bool(ok),
            "alloc_bytes": want, "pool_grew_bytes": grew}


def check_vllm(model: str, max_len: int) -> dict:  # pragma: no cover - needs vllm
    import torch  # noqa: F401
    from umallm.vllm_integration.uma_backend import patch_vllm_kv_allocation

    patched = patch_vllm_kv_allocation()
    if not patched:
        return {"name": "vllm_kv_alloc", "pass": False,
                "reason": "patch_vllm_kv_allocation() returned False "
                          "(no known KV-alloc site; allocate in kv_alloc_scope)"}
    before = _pool_live_bytes()
    from vllm import LLM
    llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=0.6,
              enforce_eager=True)
    grew = _pool_live_bytes() - before
    # A real KV pool is at least hundreds of MiB; anything tiny means the patch
    # did not catch the KV allocation.
    ok = grew > 128 * 1024 * 1024
    del llm
    return {"name": "vllm_kv_alloc", "pass": bool(ok),
            "pool_grew_bytes": grew, "model": model}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alloc-mib", type=int, default=512,
                    help="size of the direct-scope probe allocation")
    ap.add_argument("--model", default=None,
                    help="HF id for the optional vLLM KV-alloc check (e.g. "
                         "meta-llama/Llama-3.1-8B)")
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args(argv)

    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if not cuda:
        print("No CUDA device -- this verifier needs a GPU (any CUDA box, not "
              "only GH200). Skipped.")
        return 2
    if not native_available():
        print("native_available() is False: build the extension first with\n"
              "    UMA_BUILD_CUDA=1 pip install -e .[vllm]\n"
              "(and run on a device with concurrentManagedAccess).")
        return 2

    checks = [check_direct(args.alloc_mib)]
    if args.model:
        checks.append(check_vllm(args.model, args.max_len))

    ok = True
    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        ok = ok and c["pass"]
        extra = ", ".join(f"{k}={v}" for k, v in c.items()
                          if k not in ("name", "pass"))
        print(f"  [{status}] {c['name']}: {extra}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

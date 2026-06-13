# PeerKV Development Plan

This directory now keeps only product/runtime planning material. Paper-specific
planning files were removed from the code repository.

## Current Scope

PeerKV has one mainline target:

- vLLM serving runtime
- CUDA kernels
- A100/H100 NVLink multi-GPU systems
- do-no-harm runtime selection and CI coverage

MLX, GH200/UMA, and managed-memory work is parked unless it is needed for
compatibility or explicit experiments.

## Remaining Documents

| File | Purpose |
| --- | --- |
| `03_track_C_product_runtime.md` | Runtime/product implementation plan |
| `04_cross_cutting.md` | CI, invariants, observability, packaging, and risk notes |

## Engineering Rules

- Build one real runtime path rather than paper-only scaffolding.
- Keep `umallm/`, `csrc/`, `tests/`, `scripts/`, and `experiments/` aligned.
- Treat do-no-harm behavior as code and tests, not documentation only.
- Keep generated benchmark results out of git unless there is a specific reason
  to publish a small fixture.

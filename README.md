# PeerKV

PeerKV is a research runtime for multi-GPU long-context LLM inference. The
current mainline targets vLLM, CUDA, and A100/H100-class NVLink systems.

The importable Python package is still named `umallm` for compatibility.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `umallm/` | Python runtime, policies, placement logic, observability, and vLLM integration |
| `csrc/` | CUDA/C++ kernels and native benchmarks |
| `tests/` | CPU-safe unit tests plus GPU-marked tests |
| `experiments/` | Hardware probes and reproducibility scripts; generated results are not committed |
| `scripts/` | Environment setup, hardware activation, benchmark runners, and validation helpers |
| `examples/` | Minimal example launch scripts |
| `docs/` | Engineering notes and timing audit source snapshots |
| `collaboration_plan/` | Product/runtime development plan |

## Mainline Scope

The active code path is:

- `umallm/peerkv/`
- `umallm/elastic_policy.py`
- `umallm/multigpu.py`
- `umallm/peer_parallel_attn.py`
- `umallm/vllm_integration/`
- `csrc/`

Apple/MLX, GH200/UMA, and managed-memory code is retained for compatibility and
experimentation, but it is not the mainline runtime.

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e '.[torch]'
pip install -e '.[vllm]'
pip install -e '.[mlx]'
```

Build the CUDA extension when needed:

```bash
UMA_BUILD_CUDA=1 pip install -e '.[vllm]'
```

## Tests

CPU-safe tests:

```bash
pytest tests/ -m "not gpu"
```

GPU tests are marked with `@pytest.mark.gpu` and require an appropriate CUDA
machine. Experiments under `experiments/` are run directly rather than collected
as pytest tests.

## Hardware Notes

Before running NVLink-sensitive experiments on shared machines, ensure the box is
idle and the expected topology is active. The helper scripts under `scripts/`
include NVLink activation and hardware-specific runners.

```bash
bash scripts/activate_nvlink.sh
nvidia-smi topo -m
```

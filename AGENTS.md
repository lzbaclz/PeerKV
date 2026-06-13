# AGENTS.md — PeerKV contributor & agent guide

Read this before touching code. Full plan: `collaboration_plan/README.md` (single
entry point). First-week steps: `scripts/onboard.sh` then `collaboration_plan/README.md` §6.

## The three global decisions (do not violate)
- **D1 — One real system.** Track C Phase 1–2 *is* Track B's experiment chapter.
  Don't build paper-only scaffolding that won't ship in the product.
- **D2 — One mainline = vLLM + CUDA + A100/H100 NVLink.** MLX / GH200 / UMA /
  CUDA-managed-memory are **parked Track D** (phase 4 only). Mainline code is
  `umallm/peerkv/`, `umallm/elastic_policy.py`, `umallm/multigpu.py`,
  `umallm/peer_parallel_attn.py`, `umallm/vllm_integration/`, `csrc/`.
- **D3 — "Do no harm" is a coded invariant + CI test, not a doc sentence.** Three
  red lines (R1): (1) a single-GPU-fittable request is never made slower → `SINGLE`;
  (2) peer compute-busy ⇒ no CFK; (3) NVLink eff-bw < PCIe ⇒ route host.
  Spec: `collaboration_plan/04_cross_cutting.md` §1. (Direction push/pull is **not**
  a red line — Track A proved it a non-effect.)

## Scientific integrity rules (Track A, hard-won)
1. Any "X is free/faster under load" claim: check the timing wasn't drained by a
   whole-device `synchronize()`.
2. GPU microbenchmarks **lock clocks** (A100 `nvidia-smi -lgc 1410,1410`).
3. Victim/latency = ratio + per-iteration CUDA events, never whole-device sync.
4. Every perf claim needs a hardware counter (DCGM DRAM/SM/NVLink), not wall-clock alone.
5. Never quote a margin without its geometry (**MHA vs GQA** differ ~5×).

## Mainline module map
| Path | What | Status |
|---|---|---|
| `umallm/elastic_policy.py` | offline 5-corner selector (`select_point`) | works, CPU-tested |
| `umallm/peer_parallel_attn.py` | exact CFK attention math (merge/ring) | works, CPU-tested |
| `umallm/multigpu.py` | cost model + topology-aware placement | works |
| `umallm/peerkv/` | CUDA CFK fast path (JIT `csrc/peer_fused_attn_ext.cu`) | prototype |
| `umallm/vllm_integration/` | `KVConnectorBase_V1` + attention impl | prototype; **register is monkeypatch (9.A.1, to remove)** |
| `umallm/observability/` | box_probe + topology_probe (runnable); metrics (defined, unwired) | partial |
| `umallm/runtime/` | online selector/kv_manager/calibration | **PLANNED stubs** |
| `umallm/_parked` *(by convention)* `mlx_*`, `uma_*`, `grace_hopper`, `pressure`, `calibration` | Track D | **PARKED — do not extend** |

## Experiments (Track A)
`experiments/gN_*.py` are hardware probes, run directly (not pytest):
`g1` attribution · `g2` handoff window · `g3` placement cost law · `g4` measurement
trap · `g5` read-vs-write port (new). Always `assert_box_idle()` first on the shared
box. Reproduce: `scripts/reproduce_track_a.sh` (A100), `scripts/run_h100_suite.sh` (H100).

## Tests / CI
- `pytest tests/ -m "not gpu"` runs on CPU (torch/gpu tests skip). `experiments/`
  is excluded from collection (`testpaths=["tests"]`).
- GPU tests are `@pytest.mark.gpu`; markers registered in `pyproject.toml`.
- CI: `.github/workflows/ci.yml` (CPU job). GPU job → self-hosted dual-A100 (TODO).

## Commits
- Conventional, imperative, scoped: `paper: …`, `Track C: …`, `experiments: …`.
- Never commit secrets or large binaries. Don't auto-submit the paper.

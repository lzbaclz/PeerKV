# PeerKV × vLLM 0.8.5 — serving integration build spec

Env: conda `peerkv-serve` (vllm 0.8.5, torch 2.6.0+cu124, transformers 4.51.3),
driver 535 (CUDA 12.2). Box is SHARED (co-tenant churns 0–54 GB on both GPUs).
Verified hooks below are from a parallel source-investigation workflow + spot-checks
against the installed tree.

## Milestones (honest grading)
- **M1 — single-GPU KV ceiling + throughput/P50/P99.** No PeerKV code; `vllm bench`.
  GPU0-only. The *denominator* every later claim divides against. **Achievable.**
- **M2+M3 — peer-GPU KV tier (coupled).** Two-tensor KV (local cuda:0 + peer cuda:1)
  + a copy-back attention backend that stages peer blocks before the kernel.
  **Stretch (multi-day CUDA debugging).**
- **M4 — compute-follows-KV in-engine.** Future work (monolithic CUDA-graph capture
  + single-device `AttentionImpl.forward` contract). The (O,lse) merge primitive it
  needs already ships at `flash_attn.py:730`.

## KEY CORRECTION (vs the raw investigation)
"Design A" = route the KV allocation through a peer-bound `torch.cuda.MemPool` to put
KV on cuda:1. This **cannot do *fractional* placement**: a torch tensor lives on ONE
device, so a MemPool only moves the *whole* per-layer tensor to cuda:1 (that's just
"run on the other GPU", not tiering). Fractional tiering ⇒ **two tensors per layer**
(local + peer) ⇒ the attention op must read from both ⇒ the copy-back backend (M3) is
**mandatory**, not optional. So M2 and M3 are one unit.

## Verified hooks (file:line in peerkv-serve site-packages/vllm)
- KV alloc site: `v1/worker/gpu_model_runner.py:1722` (`torch.zeros(kv_cache_shape,…)`)
  inside `initialize_kv_cache` (`:1689`); `self.device` at `:89`.
- num_gpu_blocks (the ceiling): computed `executor_base.py:98 determine_num_available_blocks()`,
  stored `cache_config.num_gpu_blocks` (`llm_engine.py:432`), logged `llm_engine.py:437-438`.
- EngineArgs: `engine/arg_utils.py` — gpu_memory_utilization:244, max_model_len:221,
  max_num_seqs:251, tensor_parallel_size:230, enforce_eager:262, num_gpu_blocks_override:297.
- Bench: `vllm bench throughput` (`benchmarks/throughput.py:35 run_vllm`, `:140 run_vllm_async`),
  `vllm bench serve` (`benchmarks/serve.py`, percentiles `:44-71`).
- FlashAttn v1 backend: `v1/attention/backends/flash_attn.py` — `FlashAttentionImpl.forward` `:481`;
  `reshape_and_cache_flash` `:527`; kernel `flash_attn_varlen_func` `:572`; block_table `:560/:567`.
  **Copy-back staging window = `:536`–`:572`.**
- Online-softmax (O,lse) merge already present: `flash_attn.py:730` (`prefix_output, prefix_lse`).
- Backend registration: `VLLM_ATTENTION_BACKEND` env (`selector.py:81-154`), or
  `global_force_attn_backend()`.
- Existing seam to extend (do NOT fork site-packages): `umallm/vllm_integration/uma_backend.py`
  `patch_vllm_kv_allocation()` already wraps `gpu_model_runner.initialize_kv_cache`.

## Build order (each GPU step gated on free memory; co-tenant churns)
1. **M1**: `vllm bench throughput --model NousResearch/Llama-2-7b-hf --input-len … --output-len …
   --gpu-memory-utilization … --max-model-len … --enforce-eager`; sweep max_model_len ×
   gpu_mem_util; record num_gpu_blocks → usable context = num_gpu_blocks×16. Add
   `--tensor-parallel-size 2` for the **native TP-2 baseline** (the honest "use both GPUs"
   comparison the paper's related-work TP section now needs — real vLLM numbers).
2. **M2+M3**: new `umallm/vllm_integration/peerkv_attn.py`
   (`PeerKVFlashAttentionImpl(FlashAttentionImpl)` + `PeerKVAttentionState`): allocate a peer
   KV tensor on cuda:1 for a configurable block fraction (tag device per physical block);
   in `forward()` `:536`–`:572`, async D2D-copy peer blocks → cuda:0 scratch, remap block_table
   to scratch, call kernel unchanged. Register via `peerkv_register.py` (env var).
3. **Correctness gate**: staged vs all-local `torch.allclose(atol=1e-5)`.
4. **Serve**: `vllm bench serve` throughput + P50/P99 at a context above the M1 ceiling
   (peer-tier sustains it; single-GPU rejects), vs the M1 single-GPU and native-TP-2 numbers.

## What the paper can honestly claim
- M1: measured single-GPU KV ceiling on A100 (not modeled).
- TP-2: measured vLLM TP-2 throughput/P99 — substantiates the related-work TP baseline.
- M2+M3 (if it lands): a real vLLM peer-GPU KV residency + copy-back path with
  correctness + the NVLink-staging cost/crossover vs single-GPU above the ceiling.
- M4: future work, citing the existing (O,lse) merge primitive + the piecewise-graph blocker.

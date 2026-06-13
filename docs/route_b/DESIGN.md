# PeerKV Route B Design Summary

## Problem

Long-context decoding is often constrained by KV-cache capacity. A request that
does not fit in one GPU's HBM may still fit across multiple peer GPUs connected
by NVLink.

## Design

PeerKV treats peer GPU HBM as a first-class KV tier:

- `T0`: local HBM
- `T1`: peer HBM over NVLink
- `T2`: host memory over PCIe
- `T3`: storage-backed spill, when applicable

The runtime uses calibrated bandwidth and latency estimates to decide whether a
request should stay single-GPU, use copy-back, use compute-follows-KV, or route
through host memory.

## Compute Follows KV

For overflow decode, cold KV can remain resident on a peer GPU. Instead of
moving GB-scale KV back to the compute GPU every step, the runtime sends the
small query tensor to the peer, computes attention partials near the KV, and
returns the small online-softmax partials for exact merge.

This keeps the large data movement local to the GPU that owns the KV block and
turns the interconnect exchange into a small partial-result exchange.

## Guardrails

- If the request fits on one GPU and single-GPU execution is faster, select the
  single-GPU path.
- If peer compute is busy, avoid compute-follows-KV.
- If effective NVLink bandwidth is worse than PCIe, route through host memory.
- Keep direction-specific transfer assumptions out of the policy; push and pull
  are recorded choices, not hard correctness constraints.

## Relevant Code

- `umallm/elastic_policy.py`
- `umallm/multigpu.py`
- `umallm/peer_parallel_attn.py`
- `umallm/peerkv/`
- `umallm/vllm_integration/`
- `csrc/`

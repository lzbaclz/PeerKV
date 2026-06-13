"""M3 op-level correctness gate for the PeerKV copy-back staging primitive.

Engine-independent: proves that splitting a paged KV cache across cuda:0/cuda:1 and
staging peer blocks back via PeerKVStager is *value-preserving* -- the staged local
tensor + remapped block_table yields byte-identical KV (and identical attention) to
an all-local reference. This is the numerical-correctness claim the vLLM backend
(peerkv_attn.py) rests on; the engine-level throughput/P99 is measured on a
dedicated box (H100). Runs on any 2-GPU box with a few GB free on each.

    PYTHONPATH=. python experiments/serve/m3_staging_test.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from umallm.vllm_integration.peerkv_staging import (
    PeerKVLayout, PeerKVStager, alloc_split_kv, gather_kv)

OUT = Path(__file__).resolve().parent.parent / "results" / "serve_m3_staging.json"


def main() -> None:
    if torch.cuda.device_count() < 2:
        OUT.write_text(json.dumps({"_is_measured": False, "note": "need 2 GPUs"})); return
    torch.manual_seed(0)
    C_local, C_scratch, C_peer = 8, 16, 8
    bs, H, D = 16, 8, 128
    N = C_local + C_peer                      # total logical blocks
    lay = PeerKVLayout(C_local, C_scratch, C_peer, bs, H, D)

    # ---- reference: ALL N blocks on cuda:0 (ground truth) ----
    ref = torch.randn(2, N, bs, H, D, dtype=torch.float16, device="cuda:0")

    # ---- split into PeerKV layout ----
    local_kv, peer_kv = alloc_split_kv(lay)
    local_kv.zero_()
    local_kv[:, :C_local] = ref[:, :C_local]                  # local blocks
    peer_kv[:] = ref[:, C_local:].to("cuda:1")                # peer blocks -> cuda:1
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    # ---- a block_table mixing local + peer blocks, 2 seqs x 6 blocks ----
    # vLLM convention: ids 0..N-1; ids >= C_local are physically peer-resident
    # (peer index = id - C_local). The input block_table IS the vLLM one (no
    # re-encoding) -- this is exactly what the engine produces.
    ref_bt = torch.tensor([[0, C_local+0, 1, C_local+3, 2, C_local+7],
                           [C_local+1, 3, C_local+2, 4, C_local+5, 5]],
                          dtype=torch.int32, device="cuda:0")

    # ---- stage (peer blocks -> local scratch, block_table remapped) ----
    stager = PeerKVStager(lay)
    remap = stager.stage(local_kv, peer_kv, ref_bt)
    torch.cuda.synchronize(0)

    # ---- (1) byte-exact KV equivalence, per sequence ----
    max_abs = 0.0
    for r in range(ref_bt.shape[0]):
        ref_kv = gather_kv(ref, ref_bt[r], ref_bt.shape[1])           # from all-local
        stg_kv = gather_kv(local_kv, remap[r], ref_bt.shape[1])       # from staged
        max_abs = max(max_abs, float((ref_kv - stg_kv).abs().max()))
    kv_exact = max_abs == 0.0

    # ---- (2) attention output equivalence (SDPA over gathered KV) ----
    scale = 1.0 / (D ** 0.5)
    cos_min = 1.0
    for r in range(ref_bt.shape[0]):
        q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")
        rk = gather_kv(ref, ref_bt[r], ref_bt.shape[1])       # (2, seq, H, D)
        sk = gather_kv(local_kv, remap[r], ref_bt.shape[1])
        # -> (1, H, seq, D) for SDPA
        rK, rV = rk[0].permute(1, 0, 2).unsqueeze(0), rk[1].permute(1, 0, 2).unsqueeze(0)
        sK, sV = sk[0].permute(1, 0, 2).unsqueeze(0), sk[1].permute(1, 0, 2).unsqueeze(0)
        oref = F.scaled_dot_product_attention(q, rK, rV, scale=scale)
        ostg = F.scaled_dot_product_attention(q, sK, sV, scale=scale)
        cos = float(F.cosine_similarity(oref.reshape(-1).float(), ostg.reshape(-1).float(), dim=0))
        cos_min = min(cos_min, cos)

    res = {"_experiment": "m3_staging_test", "_is_measured": True,
           "kv_bytes_identical": kv_exact, "kv_max_abs_diff": max_abs,
           "attention_cos_min": cos_min,
           "layout": {"c_local": C_local, "c_scratch": C_scratch, "c_peer": C_peer,
                      "block_size": bs, "heads": H, "head_dim": D},
           "verdict": ("PASS: peer-split + copy-back staging is value-preserving"
                       if (kv_exact and cos_min > 0.9999) else "FAIL"),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  KV bytes identical: {kv_exact} (max_abs={max_abs})")
    print(f"  attention cos (min over seqs): {cos_min:.6f}")
    print(f"  -> {res['verdict']}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()

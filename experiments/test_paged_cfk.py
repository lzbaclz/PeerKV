"""Verify paged multi-sequence CFK gather+op vs a single-GPU full-KV reference.

Synthesizes a paged KV cache split across cuda:0 (local) and cuda:1 (peer), with a
per-sequence block_table and partial last blocks, then checks paged_cfk_decode matches
reference attention over each sequence's full KV. Run on a 2-GPU NVLink box:
  python experiments/test_paged_cfk.py
"""
from __future__ import annotations
import sys
import torch

from umallm.peerkv.fused_attn import reference_attn
from umallm.peerkv.paged import paged_cfk_decode


def cosine(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def main():
    assert torch.cuda.device_count() >= 2, "need 2 GPUs"
    torch.manual_seed(0)
    H, HKV, D = 32, 8, 128
    bs = 16
    num_seqs = 5
    peer_fraction = 0.4

    seq_lens = [777, 1024, 1600, 33, 4096]   # mix incl. partial last blocks
    max_len = max(seq_lens)
    max_blk = (max_len + bs - 1) // bs

    # global paged space: enough blocks for all seqs laid end to end
    per_seq_blocks = [(L + bs - 1) // bs for L in seq_lens]
    total_blocks = sum(per_seq_blocks)
    peer_base = total_blocks  # peer block ids start here (separate id space)

    # full KV per seq (ground truth), and the contiguous query
    qs = (torch.randn(num_seqs, H, D, device="cuda:0") * 0.1).half()
    refs = torch.empty(num_seqs, H, D, device="cuda:0", dtype=torch.half)

    # build local + peer paged caches by scattering each seq's tokens block-by-block;
    # within a sequence, send a peer_fraction tail of blocks to the peer.
    # local/peer caches sized to total blocks (simple; ids are dense here).
    local_cache = torch.zeros(2, total_blocks, bs, H, D, device="cuda:0", dtype=torch.half)
    peer_cache = torch.zeros(2, total_blocks, bs, H, D, device="cuda:1", dtype=torch.half)
    block_table = torch.zeros(num_seqs, max_blk, dtype=torch.long, device="cuda:0")

    next_local = 0
    next_peer = 0
    for i, L in enumerate(seq_lens):
        nblk = per_seq_blocks[i]
        Kf = (torch.randn(HKV, L, D, device="cuda:0") * 0.1).half()
        Vf = (torch.randn(HKV, L, D, device="cuda:0") * 0.1).half()
        refs[i] = reference_attn(qs[i], Kf, Vf)
        # decide which blocks go to peer (tail fraction)
        n_peer = int(nblk * peer_fraction)
        n_local = nblk - n_peer
        for j in range(nblk):
            valid = bs if j < nblk - 1 else (L - (nblk - 1) * bs)
            # tokens for this block, per kv-head -> [bs, H, D] (pad tail with 0)
            kblk = torch.zeros(bs, H, D, device="cuda:0", dtype=torch.half)
            vblk = torch.zeros(bs, H, D, device="cuda:0", dtype=torch.half)
            t0 = j * bs
            # broadcast HKV kv-heads to H query-head slots is NOT done here; the cache
            # stores H=HKV kv-heads (paged layout uses kv heads). Use HKV in the H slot.
            kblk[:valid, :HKV, :] = Kf[:, t0:t0 + valid, :].permute(1, 0, 2)
            vblk[:valid, :HKV, :] = Vf[:, t0:t0 + valid, :].permute(1, 0, 2)
            if j < n_local:
                gid = next_local; next_local += 1
                local_cache[0, gid] = kblk; local_cache[1, gid] = vblk
            else:
                gid = peer_base + next_peer; next_peer += 1
                peer_cache[0, gid - peer_base] = kblk.to("cuda:1")
                peer_cache[1, gid - peer_base] = vblk.to("cuda:1")
            block_table[i, j] = gid

    # the synthetic cache stores HKV kv-heads in the first HKV of the H axis; trim H->HKV
    local_cache = local_cache[:, :, :, :HKV, :].contiguous()
    peer_cache = peer_cache[:, :, :, :HKV, :].contiguous()

    seq_lens_t = torch.tensor(seq_lens, device="cuda:0")
    out = paged_cfk_decode(qs, local_cache, peer_cache, block_table, seq_lens_t, peer_base)

    ok = True
    for i, L in enumerate(seq_lens):
        c = cosine(out[i], refs[i])
        print(f"seq {i}  len={L:5d}  cos={c:.6f}")
        ok = ok and c > 0.999
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

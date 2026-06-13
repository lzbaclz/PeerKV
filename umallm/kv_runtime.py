"""Backend-agnostic end-to-end decode runtime for UMA-LLM.

This is the integration that was missing: a KV cache that an actual decode
loop drives, where the residency policy assigns tiers, cold blocks are held
in KIVI-compressed form, and reads dequantize on demand -- so attention
sees a (lossily) reconstructed full KV. It runs end-to-end on CPU with the
NumPy reference model in :mod:`umallm.backends.numpy_ref`; the MLX backend
(:mod:`umallm.mlx_backend`) is the accelerated drop-in for Apple Silicon.

The cache stores KV in fixed-size blocks. After each decision step the
policy (:class:`umallm.policy.UMAPolicy`) places each *completed* block into
a residency tier from per-block attention mass; T2/T3 blocks are quantized
in place (KIVI 4-bit) and their fp32 copy is freed, T0/T1 blocks stay
resident. ``fetch`` reconstructs the full per-layer K/V by dequantizing the
compressed blocks, which is what makes the loop end-to-end: the
compression error actually flows through the attention softmax and shows up
in the output, so fidelity is measurable rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .compression import KIVI4bit, dequantize_block, quantize_block, quantize_block_2bit
from .policy import UMAPolicy
from .pressure import PressureLevel
from .uma_model import ResidencyTier


@dataclass
class _Block:
    """One KV block: resident (fp32) or compressed (KIVI4bit)."""

    tier: int
    n_tok: int
    n_heads: int
    head_dim: int
    k: np.ndarray | None = None  # (n_tok, n_heads, head_dim) fp32 when resident
    v: np.ndarray | None = None
    cK: KIVI4bit | None = None
    cV: KIVI4bit | None = None

    def resident(self) -> bool:
        return self.tier in (int(ResidencyTier.T0_GPU_ACTIVE),
                             int(ResidencyTier.T1_CPU_ACTIVE),
                             int(ResidencyTier.T0_ANE_ACTIVE))

    def kv(self) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct (n_tok, n_heads, head_dim) K and V (dequant if cold)."""
        if self.k is not None:
            return self.k, self.v
        kf = dequantize_block(self.cK).reshape(self.n_tok, self.n_heads, self.head_dim)
        vf = dequantize_block(self.cV).reshape(self.n_tok, self.n_heads, self.head_dim)
        return kf.astype(np.float32), vf.astype(np.float32)

    def bytes_actual(self) -> int:
        if self.k is not None:
            # count as fp16-resident (2 bytes) for K and V
            return self.n_tok * self.n_heads * self.head_dim * 2 * 2
        return self.cK.nbytes() + self.cV.nbytes()

    def bytes_fp16(self) -> int:
        return self.n_tok * self.n_heads * self.head_dim * 2 * 2


class TieredKVCache:
    """Per-model KV cache with attention-driven residency tiering.

    Parameters mirror the model geometry; ``policy`` is a
    :class:`umallm.policy.UMAPolicy`. ``cold_bits`` selects the cold-tier
    representation (4 or 2). ``compress`` False makes this a no-tiering
    full-precision cache (used as the fidelity reference / B1 baseline).
    """

    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 block_size: int = 8, policy: UMAPolicy | None = None,
                 cold_bits: int = 4, compress: bool = True):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.cold_bits = cold_bits
        self.compress = compress
        self.policy = policy or UMAPolicy(n_active=4, n_sink=1, n_window=1)
        self.blocks: list[list[_Block]] = [[] for _ in range(n_layers)]
        self._open: list[dict] = [
            {"k": [], "v": []} for _ in range(n_layers)
        ]
        self.n_decode_compress = 0  # how many block-compressions happened

    # ---------------------------------------------------------------- #
    def append(self, layer: int, k_vec: np.ndarray, v_vec: np.ndarray) -> None:
        """Append one token's K/V (each (n_heads, head_dim)) to a layer."""
        buf = self._open[layer]
        buf["k"].append(np.asarray(k_vec, dtype=np.float32))
        buf["v"].append(np.asarray(v_vec, dtype=np.float32))
        if len(buf["k"]) == self.block_size:
            self._close_open(layer)

    def _close_open(self, layer: int) -> None:
        buf = self._open[layer]
        if not buf["k"]:
            return
        k = np.stack(buf["k"], axis=0)  # (n_tok, n_heads, head_dim)
        v = np.stack(buf["v"], axis=0)
        self.blocks[layer].append(_Block(
            tier=int(ResidencyTier.T0_GPU_ACTIVE), n_tok=k.shape[0],
            n_heads=self.n_heads, head_dim=self.head_dim, k=k, v=v,
        ))
        self._open[layer] = {"k": [], "v": []}

    def fetch(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Full reconstructed (seq, n_heads, head_dim) K and V for a layer."""
        ks, vs = [], []
        for blk in self.blocks[layer]:
            k, v = blk.kv()
            ks.append(k)
            vs.append(v)
        buf = self._open[layer]
        if buf["k"]:
            ks.append(np.stack(buf["k"], axis=0))
            vs.append(np.stack(buf["v"], axis=0))
        if not ks:
            empty = np.zeros((0, self.n_heads, self.head_dim), np.float32)
            return empty, empty
        return np.concatenate(ks, 0), np.concatenate(vs, 0)

    # ---------------------------------------------------------------- #
    def _quantize(self, arr: np.ndarray) -> KIVI4bit:
        flat = arr.reshape(-1, self.head_dim)
        return (quantize_block_2bit(flat) if self.cold_bits == 2
                else quantize_block(flat))

    def apply_policy(self, hotness: list[np.ndarray],
                     pressure: PressureLevel = PressureLevel.NORMAL) -> None:
        """Place each completed block into a tier from per-block hotness.

        ``hotness[layer]`` is a 1-D array of per-completed-block scores
        (e.g. attention mass). No-op when ``compress`` is False.
        """
        if not self.compress:
            return
        for layer in range(self.n_layers):
            comp = self.blocks[layer]
            if len(comp) == 0:
                continue
            scores = np.asarray(hotness[layer], dtype=np.float32).reshape(-1)
            if scores.shape[0] != len(comp):  # be robust to off-by-one
                scores = np.resize(scores, len(comp))
            tiers = self.policy.place(scores, pressure=pressure)
            for blk, t in zip(comp, tiers.tolist()):
                self._reconcile(blk, int(t))

    def _reconcile(self, blk: _Block, new_tier: int) -> None:
        cold = new_tier in (int(ResidencyTier.T2_COMPRESSED),
                            int(ResidencyTier.T3_SWAPPED))
        if cold and blk.resident():
            blk.cK = self._quantize(blk.k)
            blk.cV = self._quantize(blk.v)
            blk.k = blk.v = None
            self.n_decode_compress += 1
        elif (not cold) and (not blk.resident()):
            k, v = blk.kv()
            blk.k, blk.v = k, v
            blk.cK = blk.cV = None
        blk.tier = new_tier

    # ---------------------------------------------------------------- #
    def block_hotness_from_attn(self, attn_per_token: list[np.ndarray]
                                ) -> list[np.ndarray]:
        """Aggregate per-token attention mass into per-completed-block mass.

        ``attn_per_token[layer]`` is a 1-D array of length ``seq`` giving the
        attention weight the current query put on each past token.
        """
        out = []
        for layer in range(self.n_layers):
            a = np.asarray(attn_per_token[layer], dtype=np.float32).reshape(-1)
            n_comp = len(self.blocks[layer])
            agg = np.zeros(n_comp, dtype=np.float32)
            for b in range(n_comp):
                lo = b * self.block_size
                hi = min(lo + self.block_size, a.shape[0])
                if hi > lo:
                    agg[b] = float(a[lo:hi].sum())
            out.append(agg)
        return out

    # ---------------------------------------------------------------- #
    def footprint(self) -> dict:
        actual = fp16 = 0
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for layer in range(self.n_layers):
            for blk in self.blocks[layer]:
                actual += blk.bytes_actual()
                fp16 += blk.bytes_fp16()
                counts[blk.tier] = counts.get(blk.tier, 0) + 1
        return {
            "actual_bytes": actual,
            "fp16_bytes": fp16,
            "compression_x": (fp16 / actual) if actual else 1.0,
            "tier_block_counts": counts,
            "n_decode_compress": self.n_decode_compress,
        }


def tiered_score_sequence(model, token_ids, cache) -> np.ndarray:
    """Teacher-forced pass: feed a fixed token sequence through ``model``
    using ``cache``; return the per-position logits (n_tokens, vocab).

    Decouples the compression effect from decoding divergence: the same
    tokens go through both the reference (full) and the tiered cache, so
    logit differences are purely the cost of the residency policy.
    """
    logits_seq = []
    for pos, tok in enumerate(token_ids):
        logits, attn = model.step(int(tok), cache, pos)
        logits_seq.append(logits)
        hot = cache.block_hotness_from_attn(attn)
        cache.apply_policy(hot, PressureLevel.WARN)
    return np.stack(logits_seq, axis=0)


def greedy_decode(model, prompt_ids, n_gen, cache) -> list[int]:
    """Greedy decode ``n_gen`` tokens after ``prompt_ids`` through ``cache``."""
    out: list[int] = []
    pos = 0
    logits = None
    for tok in prompt_ids:
        logits, attn = model.step(int(tok), cache, pos)
        pos += 1
        cache.apply_policy(cache.block_hotness_from_attn(attn), PressureLevel.NORMAL)
    for _ in range(n_gen):
        nxt = int(np.argmax(logits))
        out.append(nxt)
        logits, attn = model.step(nxt, cache, pos)
        pos += 1
        cache.apply_policy(cache.block_hotness_from_attn(attn), PressureLevel.WARN)
    return out


def logit_fidelity(full: np.ndarray, tiered: np.ndarray) -> dict:
    """Compare two (n_tokens, vocab) logit matrices."""
    f = full.reshape(full.shape[0], -1)
    t = tiered.reshape(tiered.shape[0], -1)
    rel_l2 = float(np.linalg.norm(f - t) / (np.linalg.norm(f) + 1e-9))
    # mean per-step cosine similarity
    cos = []
    for i in range(f.shape[0]):
        a, b = f[i], t[i]
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    argmax_agree = float(np.mean(np.argmax(f, 1) == np.argmax(t, 1)))
    return {
        "rel_l2": rel_l2,
        "mean_cosine": float(np.mean(cos)),
        "argmax_agreement": argmax_agree,
    }

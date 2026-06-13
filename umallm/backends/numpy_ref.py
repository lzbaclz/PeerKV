"""A tiny NumPy reference transformer for end-to-end runtime testing.

This is *not* a useful language model -- the weights are random. Its job is
to exercise the full UMA-LLM decode path on CPU: per token it computes real
Q/K/V, appends K/V to the :class:`umallm.kv_runtime.TieredKVCache`, reads
the (possibly dequantized) full K/V back, runs a real attention softmax, and
produces logits. Because attention runs over the reconstructed cold blocks,
the residency policy's compression error propagates into the logits exactly
as it would for a real model -- which is what lets ``e9`` measure end-to-end
fidelity rather than assume it.

On Apple Silicon the same loop runs through MLX with real weights; this
backend is the portable, sandbox-runnable reference.
"""
from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


class RefTransformer:
    """Minimal decoder-only transformer with a per-step KV-cache interface."""

    def __init__(self, n_layers: int = 2, n_heads: int = 4, head_dim: int = 32,
                 vocab: int = 256, seed: int = 0):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.vocab = vocab
        self.d = d = n_heads * head_dim
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d)
        self.emb = (rng.normal(0, 1, (vocab, d)) * 0.1).astype(np.float32)
        self.Wq = [(rng.normal(0, 1, (d, d)) * s).astype(np.float32) for _ in range(n_layers)]
        self.Wk = [(rng.normal(0, 1, (d, d)) * s).astype(np.float32) for _ in range(n_layers)]
        self.Wv = [(rng.normal(0, 1, (d, d)) * s).astype(np.float32) for _ in range(n_layers)]
        self.Wo = [(rng.normal(0, 1, (d, d)) * s).astype(np.float32) for _ in range(n_layers)]
        self.W1 = [(rng.normal(0, 1, (d, 4 * d)) * s).astype(np.float32) for _ in range(n_layers)]
        self.W2 = [(rng.normal(0, 1, (4 * d, d)) * s).astype(np.float32) for _ in range(n_layers)]
        self.Wout = (rng.normal(0, 1, (d, vocab)) * s).astype(np.float32)
        # fixed positional vectors so identical tokens at different positions differ
        freqs = np.exp(-np.arange(0, d, 2) * (np.log(10000.0) / d))
        self._freqs = freqs.astype(np.float32)

    def _pos(self, pos: int) -> np.ndarray:
        ang = pos * self._freqs
        pe = np.zeros(self.d, dtype=np.float32)
        pe[0::2] = np.sin(ang)
        pe[1::2] = np.cos(ang)
        return 0.05 * pe

    def step(self, token_id: int, cache, pos: int):
        """One decode step. Returns (logits (vocab,), [attn_mass per layer])."""
        H, D = self.n_heads, self.head_dim
        x = self.emb[int(token_id)].astype(np.float32) + self._pos(pos)
        attn_per_layer = []
        for l in range(self.n_layers):
            q = (x @ self.Wq[l]).reshape(H, D)
            k = (x @ self.Wk[l]).reshape(H, D)
            v = (x @ self.Wv[l]).reshape(H, D)
            cache.append(l, k, v)
            K, V = cache.fetch(l)              # (T, H, D)
            scores = np.einsum("hd,thd->ht", q, K) / np.sqrt(D)  # (H, T)
            w = _softmax(scores, axis=1)        # (H, T)
            ctx = np.einsum("ht,thd->hd", w, V).reshape(self.d)  # (H,D)->(d,)
            x = x + ctx @ self.Wo[l]
            h = np.maximum(0.0, x @ self.W1[l])
            x = x + h @ self.W2[l]
            attn_per_layer.append(w.mean(axis=0))  # (T,) mean attention over heads
        logits = x @ self.Wout
        return logits.astype(np.float32), attn_per_layer

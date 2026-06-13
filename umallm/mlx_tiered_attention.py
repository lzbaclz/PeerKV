"""Phase 1: block-wise *tiered attention* for mlx-lm (the C4 systems core).

Validated in Phase 0 (`experiments/e11_tiered_attention_spike.py`): keep recent
("hot") KV blocks in fp16 and cold blocks in MLX-native low-bit, and compute
attention as a **partitioned online-softmax (flash) merge** so the full fp16
context is *never* materialized -> live peak is bounded and decode stays ~1.1x
of dense (not the 28x of the old NumPy reconstruct path).

Partitions of the keys/values, by age (StreamingLLM-style residency):
  * SINK   : first ``n_sink_blocks`` blocks, fp16, strictly-past -> unmasked
  * COLD   : the middle, low-bit (mx.quantize), strictly-past -> unmasked
  * WINDOW : most recent ``n_window_blocks`` blocks + open tail, fp16, may
             overlap the current query chunk -> causal-masked
Because we only ever demote *completed* blocks that are behind the current
position, SINK and COLD are always strictly older than any live query, so only
WINDOW needs a causal mask. This holds for both prefill (L>1) and decode (L=1).

The merge is exact (Phase-0 gate A: err 4.5e-08); the only quality cost is the
cold tier's quantization.
"""
from __future__ import annotations

import mlx.core as mx


# --------------------------------------------------------------------------- #
# partitioned online-softmax attention primitives
# --------------------------------------------------------------------------- #
def _apply_mask(scores, mask):
    if mask is None:
        return scores
    return mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype))


def _partial_fp16(q_scaled, k, v, mask=None):
    """Return (o, m, l) of attention over an fp16 partition."""
    s = q_scaled @ mx.swapaxes(k, -1, -2)
    s = _apply_mask(s, mask)
    m = mx.max(s, axis=-1, keepdims=True)
    p = mx.exp(s - m)
    l = mx.sum(p, axis=-1, keepdims=True)
    return p @ v, m, l


def _partial_quant(q_scaled, kq, vq, gs, bits, mask=None):
    """Return (o, m, l) over a low-bit partition via mx.quantized_matmul."""
    s = mx.quantized_matmul(q_scaled, *kq, transpose=True, group_size=gs, bits=bits)
    s = _apply_mask(s, mask)
    m = mx.max(s, axis=-1, keepdims=True)
    p = mx.exp(s - m)
    l = mx.sum(p, axis=-1, keepdims=True)
    o = mx.quantized_matmul(p, *vq, transpose=False, group_size=gs, bits=bits)
    return o, m, l


def _merge(parts):
    """Flash-style online-softmax combine of >=1 (o, m, l) partials (exact)."""
    parts = [p for p in parts if p is not None]
    m_all = parts[0][1]
    for _, m, _l in parts[1:]:
        m_all = mx.maximum(m_all, m)
    num = den = None
    for o, m, l in parts:
        w = mx.exp(m - m_all)
        num = o * w if num is None else num + o * w
        den = l * w if den is None else den + l * w
    return num / den


# --------------------------------------------------------------------------- #
# tiered cache
# --------------------------------------------------------------------------- #
class MLXTieredCache:
    """mlx-lm cache that tiers KV (hot fp16 + cold low-bit) and serves attention
    via the partitioned merge. Use with :func:`enable_tiered_attention`.

    Must NOT expose a ``bits`` attribute (mlx-lm would route it to its own
    all-or-nothing quantized SDPA). The tiering is *selective*, decided here.
    """

    step = 256

    def __init__(self, block_size=256, n_sink_blocks=1, n_window_blocks=8,
                 cold_bits=4, group_size=64):
        self.block_size = int(block_size)
        self.n_sink_blocks = int(n_sink_blocks)
        self.n_window_blocks = int(n_window_blocks)
        self.cold_bits = int(cold_bits)
        self.group_size = int(group_size)
        self.offset = 0
        self._geom = None              # (B, n_kv_heads, Dk, Dv, dtype)
        self._sink = []                # list[(k,v)] fp16 blocks
        self._win = []                 # list[(k,v)] fp16 blocks (recent)
        self._cold_k = None            # (wq, scales, biases) concat over tokens
        self._cold_v = None
        self._n_cold_tok = 0
        self._tail_k = None            # (B,H,t,Dk) fp16, t < block_size
        self._tail_v = None

    # ----- mlx-lm contract --------------------------------------------- #
    def update_and_fetch(self, keys, values):
        if self._geom is None:
            B, H, _, Dk = keys.shape
            self._geom = (B, H, Dk, values.shape[-1], keys.dtype)
        self._tail_k = keys if self._tail_k is None else mx.concatenate([self._tail_k, keys], axis=2)
        self._tail_v = values if self._tail_v is None else mx.concatenate([self._tail_v, values], axis=2)
        self.offset += keys.shape[2]
        while self._tail_k.shape[2] >= self.block_size:
            bk = self._tail_k[..., :self.block_size, :]
            bv = self._tail_v[..., :self.block_size, :]
            self._tail_k = self._tail_k[..., self.block_size:, :]
            self._tail_v = self._tail_v[..., self.block_size:, :]
            self._place_block(bk, bv)
        # Return value is ignored by the patched attention; hand back the new
        # chunk so an *unpatched* path at least doesn't crash on shapes.
        return keys, values

    def _place_block(self, bk, bv):
        if len(self._sink) < self.n_sink_blocks:
            self._sink.append((bk, bv))
            return
        self._win.append((bk, bv))
        if len(self._win) > self.n_window_blocks:
            ok, ov = self._win.pop(0)          # oldest window block ages out -> cold
            self._to_cold(ok, ov)

    def _to_cold(self, k, v):
        kq = mx.quantize(k, group_size=self.group_size, bits=self.cold_bits)
        vq = mx.quantize(v, group_size=self.group_size, bits=self.cold_bits)
        if self._cold_k is None:
            self._cold_k, self._cold_v = list(kq), list(vq)
        else:
            # concat each component (packed, scales, biases) along the token axis
            self._cold_k = [mx.concatenate([a, b], axis=-2) for a, b in zip(self._cold_k, kq)]
            self._cold_v = [mx.concatenate([a, b], axis=-2) for a, b in zip(self._cold_v, vq)]
        self._n_cold_tok += k.shape[2]

    def size(self):
        return self.offset

    def empty(self):
        return self.offset == 0

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0

    def make_mask(self, N, return_array=False, window_size=None):
        # attention is computed internally; expose a standard mask for any code
        # path that asks (not used by tiered_attention).
        from mlx_lm.models.cache import create_attention_mask
        return create_attention_mask(N, self.offset, return_array, window_size)

    @property
    def nbytes(self):
        tot = 0
        for grp in (self._sink, self._win):
            for k, v in grp:
                tot += k.nbytes + v.nbytes
        for cold in (self._cold_k, self._cold_v):
            if cold is not None:
                tot += sum(x.nbytes for x in cold)
        if self._tail_k is not None and self._tail_k.shape[2] > 0:
            tot += self._tail_k.nbytes + self._tail_v.nbytes
        return int(tot)

    @property
    def state(self):
        # mlx-lm's generate calls mx.eval([c.state ...]) after prefill; the
        # real stored tensors are already materialized by the attention pass,
        # so a tiny eval-safe placeholder suffices (save/load is not supported
        # for a lossy tiered cache).
        return (mx.zeros((1,), mx.float16), mx.zeros((1,), mx.float16))

    @state.setter
    def state(self, v):
        pass

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass

    # ----- diagnostics -------------------------------------------------- #
    def footprint(self):
        B, H, Dk, Dv, _ = self._geom if self._geom else (0, 0, 0, 0, None)
        actual = self.nbytes
        fp16 = int(self.offset * B * H * (Dk + Dv) * 2)
        return {
            "actual_bytes": actual, "fp16_bytes": fp16,
            "compression_x": (fp16 / actual) if actual else 1.0,
            "sink_blocks": len(self._sink), "window_blocks": len(self._win),
            "cold_tokens": self._n_cold_tok, "offset": self.offset,
        }

    # ----- the tiered attention (flash; 2-D tiled) --------------------- #
    def tiered_attention(self, queries, scale, q_tile=256, k_budget=1_048_576):
        """Flash-style tiered attention over sink/cold/window partitions.

        queries: (B, n_q_heads, L, D). Returns (B, n_q_heads, L, D).
        Assumes ``update_and_fetch`` for this chunk already ran (offset updated).

        We tile BOTH axes so the live score transient is O(q_tile x k_tile),
        independent of context length, and the cold tier stays packed:
          * outer loop over query tiles (bounds prefill);
          * inner imperative online-softmax accumulator streamed over key tiles,
            with ``mx.eval`` per tile so each tile's scores are freed before the
            next -> bounded peak (the only way, since MLX's fused SDPA does not
            expose the log-sum-exp needed to merge fp16-hot with quantized-cold).
        ``k_tile`` adapts to ``q_tile`` (k_budget / q_tile) so decode (q_tile=1)
        does the cold tier in one shot (tiny (1 x Tk) scores -> no eval churn).
        Partitions are causal-masked by absolute position (a no-op once strictly
        in the past). Exact modulo the cold tier's quantization.
        """
        B, n_q, L, D = queries.shape
        _, n_kv, Dk, Dv, dtype = self._geom
        n_rep = n_q // n_kv
        qsf = queries * scale
        if n_rep > 1:
            qsf = qsf.reshape(B, n_kv, n_rep, L, D)
        nd = qsf.ndim
        neg = mx.array(mx.finfo(mx.float32).min, mx.float32)

        def kv_dims(x):
            return mx.expand_dims(x, axis=-3) if n_rep > 1 else x

        # partition tensors, built once
        sink_k = sink_v = None
        sink_len = 0
        if self._sink:
            sink_k = kv_dims(mx.concatenate([k for k, _ in self._sink], axis=2))
            sink_v = kv_dims(mx.concatenate([v for _, v in self._sink], axis=2))
            sink_len = sink_k.shape[-2]
        cold_len = self._n_cold_tok if self._cold_k is not None else 0
        wk = [k for k, _ in self._win]
        wv = [v for _, v in self._win]
        if self._tail_k is not None and self._tail_k.shape[2] > 0:
            wk, wv = wk + [self._tail_k], wv + [self._tail_v]
        win_k = kv_dims(mx.concatenate(wk, axis=2)) if wk else None
        win_v = kv_dims(mx.concatenate(wv, axis=2)) if wk else None
        w_tok = win_k.shape[-2] if win_k is not None else 0

        def flash_qtile(qs, qa0, Lt):
            k_tile = max(self.block_size, k_budget // max(1, Lt))
            shp = qs.shape[:-1]                       # (..., Lt)
            m = mx.full((*shp, 1), float(mx.finfo(mx.float32).min), mx.float32)
            l = mx.zeros((*shp, 1), mx.float32)
            o = mx.zeros((*shp, D), mx.float32)

            def pmask(start, length):
                if Lt == 1 or start + length <= qa0:
                    return None
                qpos = mx.arange(qa0, qa0 + Lt).reshape(Lt, 1)
                kpos = mx.arange(start, start + length).reshape(1, length)
                return (qpos >= kpos).reshape((1,) * (nd - 2) + (Lt, length))

            def step(scores, vmatmul, mask):
                nonlocal m, l, o
                s = scores.astype(mx.float32)
                if mask is not None:
                    s = mx.where(mask, s, neg)
                m_new = mx.maximum(m, mx.max(s, axis=-1, keepdims=True))
                corr = mx.exp(m - m_new)
                p = mx.exp(s - m_new)
                l = l * corr + mx.sum(p, axis=-1, keepdims=True)
                o = o * corr + vmatmul(p.astype(dtype)).astype(mx.float32)
                m = m_new
                if Lt > 1:                 # prefill: free each tile's scores to
                    mx.eval(m, l, o)       # bound peak. Decode (L=1) scores are
                                           # (1 x Tk), already tiny -> no churn.

            if sink_k is not None:
                step(qs @ mx.swapaxes(sink_k, -1, -2),
                     lambda p: p @ sink_v, pmask(0, sink_len))
            if cold_len:
                for i in range(0, cold_len, k_tile):
                    ln = min(k_tile, cold_len - i)
                    kqi = [kv_dims(c[..., i:i + ln, :]) for c in self._cold_k]
                    vqi = [kv_dims(c[..., i:i + ln, :]) for c in self._cold_v]
                    step(mx.quantized_matmul(qs, *kqi, transpose=True,
                                             group_size=self.group_size,
                                             bits=self.cold_bits),
                         lambda p, vqi=vqi: mx.quantized_matmul(
                             p, *vqi, transpose=False,
                             group_size=self.group_size, bits=self.cold_bits),
                         pmask(sink_len + i, ln))
            if win_k is not None:
                step(qs @ mx.swapaxes(win_k, -1, -2),
                     lambda p: p @ win_v, pmask(sink_len + cold_len, w_tok))
            return (o / l).astype(dtype)

        chunk_start = self.offset - L
        if L <= q_tile:
            out = flash_qtile(qsf, chunk_start, L)
        else:
            out = mx.concatenate(
                [flash_qtile(qsf[..., i:i + q_tile, :], chunk_start + i,
                             min(q_tile, L - i)) for i in range(0, L, q_tile)],
                axis=-2)
        if n_rep > 1:
            out = out.reshape(B, n_q, L, D)
        return out.astype(dtype)


# --------------------------------------------------------------------------- #
# wire it into a model's attention
# --------------------------------------------------------------------------- #
def enable_tiered_attention(model):
    """Patch ``scaled_dot_product_attention`` in the model's module so any
    layer whose cache is an :class:`MLXTieredCache` uses the tiered path.

    Idempotent; returns the module that was patched. The model's attention
    keeps calling ``cache.update_and_fetch`` (which appends + tiers); the
    patched SDPA ignores the returned dense K/V and calls ``tiered_attention``.
    """
    import importlib
    modname = type(model).__module__            # e.g. mlx_lm.models.llama
    mod = importlib.import_module(modname)
    orig = getattr(mod, "scaled_dot_product_attention", None)
    if orig is None:
        raise RuntimeError(f"{modname} has no scaled_dot_product_attention to patch")
    if getattr(orig, "_uma_tiered", False):
        return mod

    def patched(queries, keys, values, cache=None, scale=1.0, mask=None, **kw):
        if isinstance(cache, MLXTieredCache):
            return cache.tiered_attention(queries, scale)
        return orig(queries, keys, values, cache=cache, scale=scale, mask=mask, **kw)

    patched._uma_tiered = True
    patched._uma_orig = orig
    mod.scaled_dot_product_attention = patched
    return mod


def disable_tiered_attention(model):
    """Undo :func:`enable_tiered_attention`."""
    import importlib
    mod = importlib.import_module(type(model).__module__)
    cur = getattr(mod, "scaled_dot_product_attention", None)
    if cur is not None and getattr(cur, "_uma_tiered", False):
        mod.scaled_dot_product_attention = cur._uma_orig
    return mod

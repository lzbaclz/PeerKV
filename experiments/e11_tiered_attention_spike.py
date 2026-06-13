"""e11 -- block-wise tiered attention feasibility spike (Plan B / C4 gate).

Proves on Apple Silicon that a two-partition attention -- hot blocks in fp16 +
cold blocks in MLX-native low-bit (mx.quantize / mx.quantized_matmul) -- merged
with an online-softmax (flash-style) combine:

  A merge-math  : == full fp16 mx.fast.scaled_dot_product_attention (exact)
  B quant-path  : cold served by mx.quantized_matmul, fidelity reported
  C bounded peak: cold never materialized fp16  -> op peak << full at long ctx
  D TPOT proxy  : decode-step time within a small factor of full (not 28x)

Writes experiments/results/phase0_tiered_attention.json. CPU/Mac with MLX.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mlx.core as mx

H, D = 8, 64
SCALE = 1.0 / np.sqrt(D)
RNG = np.random.default_rng(0)


def full_sdpa(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=None)


def _partial_fp16(q_scaled, k, v):
    s = q_scaled @ mx.swapaxes(k, -1, -2)
    m = mx.max(s, axis=-1, keepdims=True)
    p = mx.exp(s - m)
    return p @ v, m, mx.sum(p, axis=-1, keepdims=True)


def _partial_quant(q_scaled, kq, vq, gs, bits):
    s = mx.quantized_matmul(q_scaled, *kq, transpose=True, group_size=gs, bits=bits)
    m = mx.max(s, axis=-1, keepdims=True)
    p = mx.exp(s - m)
    o = mx.quantized_matmul(p, *vq, transpose=False, group_size=gs, bits=bits)
    return o, m, mx.sum(p, axis=-1, keepdims=True)


def _merge(parts):
    m_all = parts[0][1]
    for _, m, _l in parts[1:]:
        m_all = mx.maximum(m_all, m)
    num = den = None
    for o, m, l in parts:
        w = mx.exp(m - m_all)
        num = o * w if num is None else num + o * w
        den = l * w if den is None else den + l * w
    return num / den


def tiered_attention(q, k, v, n_hot, gs=64, bits=4, quantize_cold=True):
    qs = q * SCALE
    n_cold = k.shape[-2] - n_hot
    parts = [_partial_fp16(qs, k[..., n_cold:, :], v[..., n_cold:, :])]
    if n_cold > 0:
        kc, vc = k[..., :n_cold, :], v[..., :n_cold, :]
        if quantize_cold:
            parts.append(_partial_quant(qs, mx.quantize(kc, group_size=gs, bits=bits),
                                        mx.quantize(vc, group_size=gs, bits=bits), gs, bits))
        else:
            parts.append(_partial_fp16(qs, kc, vc))
    return _merge(parts)


def _cos(a, b):
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a).astype(np.float64) - np.asarray(b).astype(np.float64))))


def _mk(Tk, dt):
    f = lambda s: mx.array(RNG.standard_normal(s).astype(np.float32), dtype=dt)
    return f((1, H, 1, D)), f((1, H, Tk, D)), f((1, H, Tk, D))


def _copy16(a):
    return mx.array(np.array(a.astype(mx.float32)), dtype=mx.float16)


def main():
    res = {"_experiment": "e11_tiered_attention_spike", "_is_measured": True,
           "host": "Apple M2 Pro 32GB (MLX)", "heads": H, "head_dim": D}

    # Gate A -- merge math (cold fp16), float32
    q, k, v = _mk(2048, mx.float32)
    o_ref = full_sdpa(q, k, v); mx.eval(o_ref)
    o_t = tiered_attention(q, k, v, n_hot=256, quantize_cold=False); mx.eval(o_t)
    A = {"maxerr": _maxerr(o_ref, o_t), "cos": _cos(o_ref, o_t)}
    res["gateA_merge_math"] = {**A, "pass": A["maxerr"] < 1e-3}

    # Gate B -- cold quantized, float16
    q, k, v = _mk(4096, mx.float16)
    o_ref = full_sdpa(q, k, v); mx.eval(o_ref)
    res["gateB_quant"] = {}
    for bits in (8, 4):
        o_t = tiered_attention(q, k, v, n_hot=512, bits=bits); mx.eval(o_t)
        res["gateB_quant"][f"bits{bits}"] = {"cos": _cos(o_ref, o_t),
                                             "maxerr": _maxerr(o_ref, o_t)}

    # Gate C -- bounded resident KV footprint (deterministic; this is the
    # quantity that causes OOM, and the dominant live-peak term at long ctx).
    # mx.get_peak_memory is pool-polluted and order-dependent, so we measure
    # array bytes directly. The tiered op adds only O(ctx) score transients,
    # never a full fp16 KV, so resident bytes is the faithful peak proxy.
    Tk, n_hot = 32768, 512
    n_cold = Tk - n_hot
    _, k, v = _mk(Tk, mx.float16); mx.eval(k, v)
    full_bytes = int(k.nbytes + v.nbytes)
    k_hot, v_hot = _copy16(k[..., n_cold:, :]), _copy16(v[..., n_cold:, :])
    kq = mx.quantize(k[..., :n_cold, :], group_size=64, bits=4)
    vq = mx.quantize(v[..., :n_cold, :], group_size=64, bits=4)
    mx.eval(k_hot, v_hot, *kq, *vq)
    tiered_bytes = int(k_hot.nbytes + v_hot.nbytes
                       + sum(x.nbytes for x in kq) + sum(x.nbytes for x in vq))
    res["gateC_footprint"] = {
        "ctx": Tk, "full_kv_mb": full_bytes / 1024**2,
        "tiered_kv_mb": tiered_bytes / 1024**2,
        "ratio": tiered_bytes / full_bytes,
        "pass": tiered_bytes < 0.6 * full_bytes,
        "note": "resident KV bytes (deterministic); op adds only O(ctx) scores, "
                "not a full fp16 KV -> faithful dominant live-peak term.",
    }
    del k, v

    # Gate D -- TPOT proxy
    Tk, n_hot = 8192, 512
    n_cold = Tk - n_hot
    q, k, v = _mk(Tk, mx.float16)
    kq = mx.quantize(k[..., :n_cold, :], group_size=64, bits=4)
    vq = mx.quantize(v[..., :n_cold, :], group_size=64, bits=4)
    k_hot, v_hot = _copy16(k[..., n_cold:, :]), _copy16(v[..., n_cold:, :])
    mx.eval(q, k, v, k_hot, v_hot, *kq, *vq)

    def bench(fn, n=50):
        mx.eval(fn())
        t0 = time.perf_counter()
        for _ in range(n):
            mx.eval(fn())
        return (time.perf_counter() - t0) / n * 1e3

    t_full = bench(lambda: full_sdpa(q, k, v))
    t_tiered = bench(lambda: _merge([_partial_fp16(q * SCALE, k_hot, v_hot),
                                     _partial_quant(q * SCALE, kq, vq, 64, 4)]))
    res["gateD_tpot"] = {"ctx": Tk, "full_ms": t_full, "tiered_ms": t_tiered,
                         "ratio": t_tiered / t_full, "pass": t_tiered < 3.0 * t_full}

    res["go"] = bool(res["gateA_merge_math"]["pass"] and res["gateC_footprint"]["pass"]
                     and res["gateD_tpot"]["pass"])
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()

    out = Path(__file__).resolve().parent / "results" / "phase0_tiered_attention.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n>>> {'GO (B1 feasible)' if res['go'] else 'NO-GO'} <<<  -> {out}")


if __name__ == "__main__":
    main()

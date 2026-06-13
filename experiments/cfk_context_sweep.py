"""CFK (compute-follows-KV) speedup vs. context length -- Track B headline sweep.

Drives the standalone multi-device fused-decoder bench (build/peer_fused_decoder_bench,
built from csrc/peer_fused_decoder.cu) across a fixed full-depth model (L=32 layers,
Llama-3-8B GQA geometry) and several context lengths, parsing its stdout into one JSON.

The bench prints (per run):
  # numerics: cosine(single, peer) = <c>
  single      <ms> ms  (<us> us/layer)
  peer-eager  <ms> ms  ratio single/peer = <r>x ...
  peer-graph  <ms> ms  ratio single/peer = <r>x ...   (or "capture failed")

We record, per context: single us/layer, peer-eager and peer-graph speedup over single,
and the numerics cosine. The story: attention's share of the layer grows with context,
so CFK (which halves the KV read) wins more at long context and tends to 1.0x at short.

Run on a dual-A100/H100 NVLink box, MIG off, clocks locked, GPUs idle:
  python experiments/cfk_context_sweep.py --contexts 4096 8192 16384 32768 --trials 10
"""
from __future__ import annotations
import argparse, json, re, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "peer_fused_decoder_bench"
OUT = ROOT / "experiments" / "results" / "cfk_context_sweep.json"

RE_COS = re.compile(r"cosine\(single, peer\)\s*=\s*([0-9.]+)")
RE_SINGLE = re.compile(r"single\s+([0-9.]+)\s*ms\s*\(\s*([0-9.]+)\s*us/layer\)")
RE_EAGER = re.compile(r"peer-eager\s+([0-9.]+)\s*ms\s*ratio single/peer\s*=\s*([0-9.]+)x")
RE_GRAPH = re.compile(r"peer-graph\s+([0-9.]+)\s*ms\s*ratio single/peer\s*=\s*([0-9.]+)x")


def run_one(layers: int, ctx: int, trials: int, splits: int, timeout: float) -> dict:
    cmd = [str(BIN), str(layers), str(ctx), str(trials), str(splits)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    out = p.stdout
    if p.returncode != 0:
        return {"ctx": ctx, "error": f"rc={p.returncode}", "stderr": p.stderr[-500:], "stdout": out[-500:]}
    cos = RE_COS.search(out)
    sg = RE_SINGLE.search(out)
    eg = RE_EAGER.search(out)
    gr = RE_GRAPH.search(out)
    row = {
        "ctx": ctx, "layers": layers, "trials": trials, "splits": splits,
        "wall_s": round(dt, 1),
        "cosine_single_vs_peer": float(cos.group(1)) if cos else None,
        "single_ms": float(sg.group(1)) if sg else None,
        "single_us_per_layer": float(sg.group(2)) if sg else None,
        "peer_eager_speedup": float(eg.group(2)) if eg else None,
        "peer_graph_speedup": float(gr.group(2)) if gr else None,
        "graph_captured": gr is not None,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--splits", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    assert BIN.exists(), f"bench binary missing: {BIN} (build via: cd csrc && make full)"

    rows = []
    for ctx in args.contexts:
        print(f"=== L={args.layers} ctx={ctx} trials={args.trials} ===", flush=True)
        try:
            row = run_one(args.layers, ctx, args.trials, args.splits, args.timeout)
        except subprocess.TimeoutExpired:
            row = {"ctx": ctx, "error": "timeout"}
        rows.append(row)
        if "error" in row:
            print(f"  ctx={ctx} ERROR: {row['error']}", flush=True)
        else:
            print(f"  single={row['single_us_per_layer']}us/L  "
                  f"eager={row['peer_eager_speedup']}x  graph={row['peer_graph_speedup']}x  "
                  f"cos={row['cosine_single_vs_peer']}", flush=True)

    res = {
        "_experiment": "cfk_context_sweep",
        "_is_measured": True,
        "geometry": "Llama-3-8B GQA (random weights)",
        "device": "dual-GPU NVLink (see host)",
        "note": ("CFK fused multi-device decoder speedup vs single-GPU full-KV, by context. "
                 "Attention share grows with context -> CFK (half-KV read) wins more at long ctx."),
        "by_context": rows,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

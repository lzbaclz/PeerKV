"""e2e load generator + TPOT meter for a running vLLM OpenAI server.

Fires C concurrent streaming completion requests in a closed loop for --secs, and
records inter-token latency (TPOT) from the server's streamed tokens. Reports
TPOT P50/P95/P99 and throughput. Used to compare a real continuous-batching vLLM
decoder under {idle, push, pull, host} cross-GPU handoff contention.

No vLLM/torch import needed here; talks HTTP via the stdlib (urllib), so it runs
under any Python (e.g. the peerkv env) regardless of the vLLM venv.
"""
from __future__ import annotations
import argparse, json, random, statistics, threading, time
from urllib import request as urlreq


def stream_one(base, model, prompt, max_tokens, tpots, lock, stop_at, unique=False):
    rng = random.Random(threading.get_ident())
    while time.time() < stop_at:
        # A unique per-request prefix defeats vLLM prefix caching, so every request
        # carries its OWN long KV cache -- this is what makes the holder's decode
        # genuinely HBM-read-bound (the point of the memory-bound e2e), instead of all
        # requests sharing one cached prefix.
        p = (f"[req-{rng.getrandbits(48):x}] " + prompt) if unique else prompt
        body = json.dumps({"model": model, "prompt": p, "max_tokens": max_tokens,
                           "temperature": 0.0, "stream": True}).encode()
        req = urlreq.Request(base + "/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
        try:
            last = None
            with urlreq.urlopen(req, timeout=60) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    now = time.perf_counter()
                    if last is not None:
                        with lock:
                            tpots.append((now - last) * 1e3)  # ms/token
                    last = now
        except Exception:
            time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--secs", type=float, default=30.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompt-words", type=int, default=0,
                    help="if >0, build a long prompt of ~N words (memory-bound holder)")
    ap.add_argument("--unique-prefix", action="store_true",
                    help="prepend a unique per-request id to defeat prefix caching")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base_sentence = ("Summarize the history and architecture of high-bandwidth memory in "
                     "modern GPUs, then explain NVLink and its role in KV cache transfer. ")
    if args.prompt_words > 0:
        reps = max(1, args.prompt_words // len(base_sentence.split()))
        prompt = base_sentence * reps
    else:
        prompt = base_sentence * 8
    tpots, lock = [], threading.Lock()
    stop_at = time.time() + args.secs
    threads = [threading.Thread(target=stream_one,
               args=(args.base, args.model, prompt, args.max_tokens, tpots, lock,
                     stop_at, args.unique_prefix))
               for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not tpots:
        print(f"[{args.label}] no tokens recorded"); return
    tpots.sort()
    def pct(p):
        return tpots[min(len(tpots) - 1, int(p * len(tpots)))]
    res = {"label": args.label, "model": args.model, "n_tokens": len(tpots),
           "tpot_ms_p50": round(statistics.median(tpots), 3),
           "tpot_ms_p95": round(pct(0.95), 3), "tpot_ms_p99": round(pct(0.99), 3),
           "tpot_ms_mean": round(statistics.fmean(tpots), 3),
           "throughput_tok_s": round(len(tpots) / args.secs, 1),
           "concurrency": args.concurrency}
    print(f"[{args.label}] tok={res['n_tokens']} TPOT p50={res['tpot_ms_p50']} "
          f"p95={res['tpot_ms_p95']} p99={res['tpot_ms_p99']} ms  "
          f"thr={res['throughput_tok_s']} tok/s")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

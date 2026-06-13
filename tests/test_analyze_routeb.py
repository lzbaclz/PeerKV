"""CPU tests for the Route B analysis/plotting pipeline."""
import csv
import json

from experiments.analyze_routeb import (
    HAS_MPL, analyze, markdown_table, table_rows, write_csv,
)


def _payload():
    """Two contexts x two modes, with hbm_only OOMing at high context."""
    def row(mode, ctx, n_ok, p50, p99, good, hbm, grace=0.0):
        return {"mode": mode, "context_tokens": ctx, "concurrency": 16,
                "n_ok": n_ok, "n_requests": 32, "tpot_p50_ms": p50,
                "tpot_p99_ms": p99, "goodput_tok_s": good,
                "hbm_high_water_gib": hbm, "grace_resident_gib": grace}
    return {
        "backend": "mock", "measured": False, "modes": ["routeb", "hbm_only"],
        "results": [
            row("routeb", 4096, 32, 12.9, 13.3, 2298, 8.0),
            row("hbm_only", 4096, 32, 12.9, 13.3, 2297, 8.0),
            row("routeb", 65536, 32, 14.2, 14.6, 1288, 40.0, grace=97.0),
            row("hbm_only", 65536, 5, 12.8, 13.1, 214, 40.0),
        ],
    }


def test_table_rows_sorted_and_served_fraction():
    rows = table_rows(_payload())
    # sorted by context then mode order (hbm_only before routeb in _MODE_ORDER)
    assert rows[0]["context_tokens"] == 4096
    assert [r["mode"] for r in rows[:2]] == ["hbm_only", "routeb"]
    hbm_hi = [r for r in rows if r["mode"] == "hbm_only"
              and r["context_tokens"] == 65536][0]
    assert hbm_hi["served"] == "5/32"
    assert abs(hbm_hi["served_frac"] - 5 / 32) < 1e-9


def test_markdown_table_has_stamp_and_rows():
    md = markdown_table(_payload())
    assert "MOCK (synthetic)" in md          # not mistaken for real data
    assert "Route B (ours)" not in md         # md table uses raw mode names
    assert "routeb" in md and "goodput" in md
    assert md.count("|") > 20                 # a real table


def test_write_csv_roundtrips(tmp_path):
    p = tmp_path / "summary.csv"
    write_csv(_payload(), str(p))
    with open(p) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert {"mode", "goodput_tok_s", "served_frac"} <= set(rows[0].keys())


def test_analyze_end_to_end(tmp_path):
    src = tmp_path / "res.json"
    src.write_text(json.dumps(_payload()))
    out = analyze(str(src), str(tmp_path / "figs"))
    assert out["measured"] is False
    assert (tmp_path / "figs" / "summary.md").exists()
    assert (tmp_path / "figs" / "summary.csv").exists()
    if HAS_MPL:
        # the four figures should exist
        assert len(out["figures"]) == 4
        for f in out["figures"]:
            assert f.endswith(".pdf")
            import os
            assert os.path.getsize(f) > 0
    else:
        assert out["figures"] == []

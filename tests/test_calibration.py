"""Tests for calibration probes."""
import json

from umallm.calibration import (
    probe_kivi_kernel,
    probe_l2_miss_latency,
    probe_soc_bandwidth,
    write_calibration,
)


def test_probes_return_positive():
    bw = probe_soc_bandwidth(buf_mb=8, n_iters=4)
    assert bw["bandwidth_gbps"] > 0
    lm = probe_l2_miss_latency(stride_bytes=128, n_lines=512)
    assert lm["per_line_ns"] > 0
    kv = probe_kivi_kernel(n_blocks=8)
    assert kv["median_us"] > 0
    assert kv["us_per_kb"] > 0


def test_write_calibration(tmp_path):
    p = tmp_path / "cal.json"
    out = write_calibration(str(p))
    assert "bandwidth" in out and "l2_miss" in out and "kivi" in out
    loaded = json.loads(p.read_text())
    assert loaded.keys() == out.keys()


def test_storage_mode_mapping():
    from umallm.metal_storage import MetalStorageMode, storage_mode_for
    from umallm.uma_model import ResidencyTier
    for t in [ResidencyTier.T0_GPU_ACTIVE, ResidencyTier.T1_CPU_ACTIVE,
              ResidencyTier.T2_COMPRESSED]:
        assert storage_mode_for(t) == MetalStorageMode.SHARED

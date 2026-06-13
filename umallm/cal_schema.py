"""Versioned calibration JSON schema — addresses 100-round R32 (KAIST)."""
from __future__ import annotations

CAL_SCHEMA_VERSION = 2

CAL_SCHEMA = {
    "version": CAL_SCHEMA_VERSION,
    "fields": {
        "soc_model": {"type": "string", "example": "M2_Max"},
        "soc_bw_gbps": {"type": "float", "unit": "GB/s"},
        "l2_miss_ns": {"type": "float", "unit": "ns/line"},
        "l2_line_bytes": {"type": "int"},
        "compress_us_per_kb": {"type": "float"},
        "decompress_us_per_kb": {"type": "float"},
        "swap_in_us_per_kb": {"type": "float"},
        "swap_out_us_per_kb": {"type": "float"},
        "captured_at": {"type": "iso8601"},
        "thermal_state": {"type": "string",
                           "values": ["cool", "warm", "hot", "throttled"]},
    },
    "required": ["version", "soc_model", "soc_bw_gbps", "l2_miss_ns"],
}


def validate(d: dict) -> list[str]:
    errors = []
    if d.get("version") != CAL_SCHEMA_VERSION:
        errors.append(f"version mismatch: got {d.get('version')}, expected {CAL_SCHEMA_VERSION}")
    for field_name in CAL_SCHEMA["required"]:
        if field_name not in d:
            errors.append(f"required field missing: {field_name}")
    return errors


def upgrade_v1_to_v2(d_v1: dict) -> dict:
    """Old v1 had no soc_model or thermal_state; assume defaults."""
    d = dict(d_v1)
    d["version"] = 2
    d.setdefault("soc_model", "unknown")
    d.setdefault("thermal_state", "cool")
    return d

"""Startup topology self-check (04_cross_cutting.md SS4.3 / risk 9.A.4).

Read-only probe answering: is MIG off? is NVLink trained to NV12? is P2P usable?
It does NOT mutate the system (no `nvidia-smi -mig 0`, no module reload -- those
are root ops in ``scripts/activate_nvlink.sh``); on a bad topology it points the
operator at that script and reports a degraded mode. The result is the data
source for the R1-route do-no-harm rule (NVLink<PCIe => route host).

Pure subprocess + stdlib; with no ``nvidia-smi`` it returns a skipped result so
CPU CI is unaffected.
"""
from __future__ import annotations

import re
import shutil
import subprocess


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True,
                              text=True, timeout=20).stdout
    except Exception:  # pragma: no cover
        return ""


def probe_topology(devices: tuple[int, ...] = (0, 1)) -> dict:
    """Return a dict describing MIG/NVLink/P2P state and a runtime recommendation.

    Keys: ``mig_enabled`` (bool|None), ``nvlink_label`` (e.g. "NV12"|None),
    ``peer_mode_ok`` (bool), ``mode`` ("peer"|"single-only"|"host-only"|"unknown"),
    ``advice`` (operator action when degraded).
    """
    if shutil.which("nvidia-smi") is None:
        return {"skipped": "nvidia-smi not found", "peer_mode_ok": False, "mode": "unknown"}

    mig_raw = _run(["--query-gpu=mig.mode.current", "--format=csv,noheader"])
    mig_lines = [l.strip() for l in mig_raw.splitlines() if l.strip()]
    mig_enabled = any("enable" in l.lower() for l in mig_lines) if mig_lines else None

    topo = _run(["topo", "-m"])
    m = re.search(r"\bNV(\d+)\b", topo)
    nvlink_label = f"NV{m.group(1)}" if m else None

    res = {"mig_enabled": mig_enabled, "nvlink_label": nvlink_label, "raw_mig": mig_lines}

    if mig_enabled:
        res.update(peer_mode_ok=False, mode="single-only",
                   advice="MIG is ON (it disables NVLink on A100). Run: "
                          "sudo bash scripts/activate_nvlink.sh, then re-probe.")
    elif nvlink_label and nvlink_label != "NV0":
        res.update(peer_mode_ok=True, mode="peer",
                   advice=None)
    else:
        res.update(peer_mode_ok=False, mode="host-only",
                   advice="No NVLink (NV#) in `nvidia-smi topo -m`; peer path will "
                          "route to host (R1-route). Check `scripts/activate_nvlink.sh`.")
    return res


if __name__ == "__main__":
    import json
    print(json.dumps(probe_topology(), indent=2))

#!/usr/bin/env bash
# Try to bring up direct NVLink between the two A100-SXM4 GPUs, and definitively
# diagnose whether it is even possible on this board.
#
# IMPORTANT: this box has NO NVSwitch (lspci shows no 10de:1af1; /proc/driver/
# nvidia-nvswitch/devices is empty). So nvidia-fabricmanager is NOT the lever
# here -- FM only manages NVSwitch. For 2x SXM4 with *direct* NVLink the links
# auto-train at driver init; they are inactive, so we test wiring/training.
#
# *** ROOT CAUSE FOUND 2026-06-03: the links were inactive because GPU 1 was in
# *** MIG mode. MIG DISABLES NVLink on A100. Disabling MIG (+ a driver module
# *** reload) trains all 12 links -> topo flips NODE -> NV12, peer BW 3.6 -> 273
# *** GB/s. The 2x A100-SXM4 ARE wired with a direct NVLink bridge (no NVSwitch
# *** needed); the earlier "board has no NVLink" verdict was an MIG artifact.
# *** This script now checks MIG FIRST and disables it before retraining.
#
# Run with root:  sudo bash scripts/activate_nvlink.sh
set -uo pipefail

say(){ printf '\n=== %s ===\n' "$*"; }

if [ "$(id -u)" -ne 0 ]; then
  echo "!! Need root. Re-run: sudo bash $0"; exit 1
fi

say "0. Before: link status + topology + MIG mode"
nvidia-smi nvlink -s 2>&1 | sed 's/^/   /'
nvidia-smi topo -m 2>&1 | head -4 | sed 's/^/   /'
nvidia-smi --query-gpu=index,mig.mode.current --format=csv 2>&1 | sed 's/^/   /'

say "0b. THE LEVER ON THIS BOX: disable MIG (MIG mode disables NVLink on A100)"
if nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | grep -qi enabled; then
  echo "   MIG is ENABLED on at least one GPU -> this is almost certainly why NVLink is down."
  echo "   Disabling MIG on all GPUs (needs them idle; stops persistence + frees handles)."
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . \
    && { echo "   GPUs BUSY -> free them first, then re-run."; } \
    || { systemctl stop nvidia-persistenced 2>/dev/null
         nvidia-smi -pm 0 >/dev/null 2>&1
         rmmod nvidia_drm 2>/dev/null; rmmod nvidia_modeset 2>/dev/null   # release DRM handle
         nvidia-smi -mig 0 2>&1 | sed 's/^/   /'
         nvidia-smi -pm 1 >/dev/null 2>&1; }
else
  echo "   MIG already disabled on all GPUs -- if NVLink is still down, continue to wiring tests."
fi

say "1. Kernel log: did the driver try to train NVLink? (the decisive evidence)"
# 'link not connected' / 'is not supported' => not wired. training errors => wired but failing.
dmesg 2>/dev/null | grep -iE "nvlink|nvswitch|nvrm.*link" | tail -25 | sed 's/^/   /' \
  || echo "   (nothing logged about nvlink)"

say "2. Enable persistence mode (keeps driver resident; re-inits GPUs)"
nvidia-smi -pm 1 2>&1 | sed 's/^/   /'

say "3. Reload the NVIDIA kernel modules to re-trigger link training"
echo "   (disruptive: kills any running GPU work. Skipping if GPUs are busy.)"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "   GPUs BUSY -> NOT reloading modules. Free them and re-run if needed."
else
  rmmod nvidia_uvm 2>/dev/null; rmmod nvidia_drm 2>/dev/null
  rmmod nvidia_modeset 2>/dev/null; rmmod nvidia 2>/dev/null
  modprobe nvidia 2>&1 | sed 's/^/   /'
  nvidia-smi -pm 1 >/dev/null 2>&1
  echo "   modules reloaded"
fi

say "4. After: link status + topology + P2P"
nvidia-smi nvlink -s 2>&1 | sed 's/^/   /'
nvidia-smi topo -m 2>&1 | head -4 | sed 's/^/   /'

say "VERDICT"
if nvidia-smi topo -m 2>&1 | grep -qE "NV[0-9]"; then
  echo "   SUCCESS: NVLink is ACTIVE (topo shows NV#). Re-run experiments/e15."
else
  echo "   STILL INACTIVE over NVLink. Check step-1 dmesg lines:"
  echo "     * 'link not connected' / no nvlink lines  -> the board does NOT wire"
  echo "       inter-GPU NVLink (PCIe-only carrier). Cannot be enabled in software."
  echo "     * training/handshake errors               -> wired but failing; check"
  echo "       BIOS NVLink option, reseat, or vendor firmware."
  echo "   There is NO NVSwitch here, so fabric-manager will NOT help."
fi

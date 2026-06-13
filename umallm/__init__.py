"""umallm -- the PeerKV package (imported under the legacy name ``umallm``).

MAINLINE (Track C: vLLM + CUDA + A100/H100 NVLink) -- the active project:
  ``umallm.peerkv``, ``elastic_policy``, ``multigpu``, ``peer_parallel_attn``,
  ``vllm_integration``, ``observability``, ``runtime``.

PARKED (Track D: Apple-Silicon / GH200 / UMA) -- NOT the mainline, kept for
reference only, **do not extend** (see AGENTS.md D2): ``uma_model``, ``policy``,
``pressure``, ``grace_hopper``, ``kv_runtime``, ``uma_alloc``, ``mlx_*``, and the
MLX ``calibration``. They are still re-exported below for backward-compatibility
with existing Track-D tests; a future cleanup may move them under ``_parked/``.
"""

# --- PARKED Track D (Apple-Silicon/GH200/UMA; re-exported for back-compat; do NOT extend) ---
from .uma_model import (
    UMACostModel,
    ResidencyTier,
    DeviceClass,
    SizingResult,
    KNOWN_DEVICES,
)
from .compression import (
    KIVI4bit,
    quantize_block,
    quantize_block_2bit,
    dequantize_block,
    compression_error,
)
from .policy import UMAPolicy
from .pressure import MemoryPressureListener, MockPressureListener, PressureLevel
from .grace_hopper import GraceHopperCostModel
from .kv_runtime import (
    TieredKVCache,
    tiered_score_sequence,
    greedy_decode,
    logit_fidelity,
)
from .uma_alloc import (
    UMAManagedAllocator,
    UMAResidencyController,
    device_name,
    native_available,
    regime,
)
# --- MAINLINE (Track C: CUDA / A100-H100 NVLink) ---
from .multigpu import (
    MGTier,
    MultiGPUDevice,
    MultiGPUKVModel,
    KNOWN_MULTIGPU,
    topology_aware_placement,
    capacities_in_blocks,
    estimate_decode_step_us,
)

__all__ = [
    "UMACostModel",
    "ResidencyTier",
    "DeviceClass",
    "SizingResult",
    "KNOWN_DEVICES",
    "KIVI4bit",
    "quantize_block",
    "quantize_block_2bit",
    "dequantize_block",
    "compression_error",
    "UMAPolicy",
    "MemoryPressureListener",
    "MockPressureListener",
    "PressureLevel",
    "GraceHopperCostModel",
    "TieredKVCache",
    "tiered_score_sequence",
    "greedy_decode",
    "logit_fidelity",
    "UMAManagedAllocator",
    "UMAResidencyController",
    "native_available",
    "regime",
    "device_name",
    "MGTier",
    "MultiGPUDevice",
    "MultiGPUKVModel",
    "KNOWN_MULTIGPU",
    "topology_aware_placement",
    "capacities_in_blocks",
    "estimate_decode_step_us",
]
__version__ = "0.3.0"

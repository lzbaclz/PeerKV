"""CoreML adapter stub — addresses 100-round R41 (Apple internal ML).

For models exported through CoreML, the runtime is opaque to us — we
can't intercept the attention path. This stub documents the integration
boundary and provides a forward-pass tracer for offline calibration.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoreMLAdapter:
    """Documents where UMA-LLM can/cannot reach into a CoreML-served model.

    UMA-LLM operates at the MLX / `transformers` layer. CoreML compiles to
    a closed graph executed by the OS framework; tier management at the
    KV-block level is therefore *not* supported when the model is
    deployed via CoreML.

    What we *can* do: offline-profile a CoreML model's attention pattern
    via instruments(1) + ANE_dump, then export the calibration JSON for
    use when the same model is later served via MLX.
    """
    model_path: str

    def supports_tier_management(self) -> bool:
        return False  # CoreML graph is sealed

    def offline_profile(self, prompts: list[str]) -> dict:
        """Stub: would invoke `instruments -t "Metal System Trace"` and
        parse the output."""
        return {
            "status": "not_implemented",
            "next_step": "use mlx-lm directly for serving; CoreML is for inference-only"
        }

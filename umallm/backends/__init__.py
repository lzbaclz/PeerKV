"""Compute backends for the UMA-LLM decode runtime.

``numpy_ref`` is the portable, CPU-runnable reference used for end-to-end
testing in any environment. On Apple Silicon the MLX path
(:mod:`umallm.mlx_backend`) provides the accelerated implementation.
"""

from .numpy_ref import RefTransformer

__all__ = ["RefTransformer"]

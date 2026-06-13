"""Iteration 2 — Per-head priority profile (PowerInfer-2 style).

For each attention head, compute a *priority* score offline by measuring
its attention entropy over a calibration trace. Low-entropy heads
(focused, predictable) get higher priority — their attended blocks go
to T0 first; high-entropy heads' blocks are demotable to T2 more freely.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class HeadPriorityProfile:
    """Per-(layer, head) priority in [0, 1]; higher = hotter (more
    deserving of T0 placement)."""
    priorities: np.ndarray  # shape (n_layers, n_heads)

    @classmethod
    def from_attention_traces(cls, attn_traces: list[np.ndarray]) -> "HeadPriorityProfile":
        """attn_traces: list of (n_layers, n_heads, seq_len) softmax weights
        from a calibration run. Lower entropy ⇒ higher priority.
        """
        # Stack and average entropy across the trace samples
        entropies = []
        for w in attn_traces:
            # entropy over the last dim
            p = w / (w.sum(axis=-1, keepdims=True) + 1e-9)
            e = -(p * np.log(p + 1e-9)).sum(axis=-1)  # (n_layers, n_heads)
            entropies.append(e)
        avg = np.stack(entropies, axis=0).mean(axis=0)  # (n_layers, n_heads)
        # Invert (low entropy → high priority); rescale to [0, 1]
        priors = -avg
        priors = priors - priors.min()
        priors = priors / (priors.max() + 1e-9)
        return cls(priorities=priors.astype(np.float32))

    def head_priority(self, layer: int, head: int) -> float:
        return float(self.priorities[layer, head])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "shape": list(self.priorities.shape),
            "values": self.priorities.flatten().tolist(),
        }))

    @classmethod
    def load(cls, path: str | Path) -> "HeadPriorityProfile":
        obj = json.loads(Path(path).read_text())
        arr = np.asarray(obj["values"], dtype=np.float32).reshape(obj["shape"])
        return cls(priorities=arr)

    def hot_head_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Returns (n_layers, n_heads) bool mask of hot heads."""
        return self.priorities >= threshold


def head_aware_score(
    base_score: np.ndarray, head_priors: np.ndarray, head_assignment: np.ndarray
) -> np.ndarray:
    """Apply per-head priorities to per-block scores.

    Args:
        base_score: (B,) hotness from the within-layer predictor
        head_priors: (n_heads,) per-head priorities for the current layer
        head_assignment: (B,) the head id that most attends to each block

    Returns the head-weighted score `base * head_priors[head_assignment]`.
    """
    base_score = np.asarray(base_score, dtype=np.float32)
    head_assignment = np.asarray(head_assignment, dtype=np.int64)
    head_priors = np.asarray(head_priors, dtype=np.float32)
    weights = head_priors[head_assignment]
    return (base_score * weights).astype(np.float32)

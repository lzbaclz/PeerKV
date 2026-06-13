"""Iteration 1 — LLM-in-a-Flash-style sparse predictor for promotion.

For the T2 → T0 promotion decision, we don't decompress every block on
every step. Instead, we predict which blocks the model will attend to in
the next K steps and only decompress those.

Predictor: simple closed-form using attention history at the previous
step + recency + per-head priority (from iter2_head_profile).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NextStepHotPredictor:
    """Predicts which blocks will be hot in the next K decode steps.

    Output: a (B,) probability that block b will appear in top-r in any
    of the next K steps. We use a simple closed form: pick the higher
    of (last-step attention) and (recency × prior hotness).
    """
    horizon: int = 4
    alpha: float = 0.7   # weight on observed attention vs recency

    def predict(self, ema_attention: np.ndarray, last_used: np.ndarray,
                step: int, window: float = 32.0) -> np.ndarray:
        """Returns (B,) probabilities in [0, 1] for next-K-step hotness."""
        ema_attention = np.asarray(ema_attention, dtype=np.float32)
        last_used = np.asarray(last_used, dtype=np.float32)
        a = ema_attention / (ema_attention.max() + 1e-9)
        rec = np.exp(-(step - last_used).clip(0) / window)
        return (self.alpha * a + (1 - self.alpha) * rec).astype(np.float32)

    def select_for_promotion(self, ema_attention: np.ndarray,
                             last_used: np.ndarray, step: int,
                             top_k: int) -> np.ndarray:
        """Returns the indices (≤ top_k) that should be promoted T2 → T0."""
        scores = self.predict(ema_attention, last_used, step)
        n = scores.shape[0]
        k = min(top_k, n)
        idx = np.argpartition(-scores, kth=k - 1)[:k] if k > 0 else np.array([], dtype=np.int64)
        return idx

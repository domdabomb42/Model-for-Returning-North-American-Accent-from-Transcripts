from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextLinearDecisionModel:
    """Lightweight linear classifier with sklearn-like decision_function."""

    weights: np.ndarray
    bias: np.ndarray
    classes_: np.ndarray

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32)
        self.bias = np.asarray(self.bias, dtype=np.float32)
        self.classes_ = np.asarray(self.classes_, dtype=np.int64)

    def decision_function(self, X) -> np.ndarray:
        scores = X @ self.weights
        scores = np.asarray(scores, dtype=np.float32)
        scores = scores + self.bias.reshape(1, -1)
        return scores

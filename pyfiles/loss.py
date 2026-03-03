"""Loss function definitions."""

from __future__ import annotations

import numpy as np


class CrossEntropy:
    """Numerically stable softmax cross-entropy for integer class labels."""

    def __init__(self) -> None:
        self._probs: np.ndarray | None = None
        self._y_true: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute mean cross-entropy from logits.

        Args:
            y_pred: Logits with shape (N, C).
            y_true: Integer class labels with shape (N,).
        """
        if y_pred.ndim != 2:
            raise ValueError(f"y_pred must have shape (N, C), got {y_pred.shape}.")
        if y_true.ndim != 1:
            raise ValueError(f"y_true must have shape (N,), got {y_true.shape}.")
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Batch size mismatch: y_pred has {y_pred.shape[0]}, y_true has {y_true.shape[0]}."
            )

        logits = y_pred.astype(np.float32, copy=False)
        labels = y_true.astype(np.int64, copy=False)
        n = logits.shape[0]

        if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("y_true contains class index out of range.")

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        # Add epsilon to avoid log(0) due to floating point underflow.
        eps = 1e-12
        loss = -np.mean(np.log(probs[np.arange(n), labels] + eps))

        self._probs = probs
        self._y_true = labels
        return float(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Gradient of mean softmax cross-entropy with respect to logits.

        Args:
            y_pred: Logits with shape (N, C). Used if no cached forward pass exists.
            y_true: Integer class labels with shape (N,). Used if no cached forward pass exists.
        """
        if self._probs is None or self._y_true is None:
            # Fallback path keeps API robust if backward is called directly.
            _ = self.forward(y_pred, y_true)

        assert self._probs is not None
        assert self._y_true is not None

        n = self._probs.shape[0]
        grad = self._probs.copy()
        grad[np.arange(n), self._y_true] -= 1.0
        grad /= n
        return grad

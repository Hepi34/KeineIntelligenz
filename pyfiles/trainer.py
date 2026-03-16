# Hepi34, onatic07, fritziii, 2026-03-06

"""Training loop orchestration."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

import numpy as np

from loss import CrossEntropy
from model import CNNModel
from optimizers import Optimizer


@dataclass
class TrainConfig:
    """Configuration for model training."""

    epochs: int = 5
    batch_size: int = 64
    shuffle: bool = True
    num_threads: int | None = None


class Trainer:
    """Handles fit/evaluate loops for the CNN model."""

    def __init__(
        self,
        model: CNNModel,
        loss_fn: CrossEntropy,
        optimizer: Optimizer,
        config: TrainConfig,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config
        self._set_num_threads(self.config.num_threads)

    @staticmethod
    def _set_num_threads(num_threads: int | None) -> None:
        """
        Best-effort threading control for NumPy-backed BLAS/OpenMP libs.

        Set this before heavy computation for most consistent behavior.
        """
        if num_threads is None:
            return
        if num_threads <= 0:
            raise ValueError(f"num_threads must be > 0, got {num_threads}.")

        thread_count = str(num_threads)
        os.environ["OMP_NUM_THREADS"] = thread_count
        os.environ["OPENBLAS_NUM_THREADS"] = thread_count
        os.environ["MKL_NUM_THREADS"] = thread_count
        os.environ["VECLIB_MAXIMUM_THREADS"] = thread_count
        os.environ["NUMEXPR_NUM_THREADS"] = thread_count

    @staticmethod
    def _batch_iterator(
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Feature/label size mismatch: {x.shape[0]} vs {y.shape[0]}.")
        n = x.shape[0]
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield x[batch_idx], y[batch_idx]

    @staticmethod
    def _accuracy_from_logits(logits: np.ndarray, y_true: np.ndarray) -> float:
        pred = np.argmax(logits, axis=1)
        return float(np.mean(pred == y_true))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        if self.config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.config.batch_size}.")
        if self.config.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.config.epochs}.")
        if x_val is None or y_val is None:
            raise ValueError("x_val and y_val are required to track test accuracy each epoch.")

        history: dict[str, list[float]] = {"loss": [], "accuracy": [], "epoch_time": []}

        for _ in range(self.config.epochs):
            epoch_start = time.perf_counter()

            train_metrics = self.train_epoch(x_train, y_train)
            eval_metrics = self.evaluate(x_val, y_val)

            epoch_time = time.perf_counter() - epoch_start
            history["loss"].append(float(train_metrics["loss"]))
            history["accuracy"].append(float(eval_metrics["accuracy"]))
            history["epoch_time"].append(float(epoch_time))

        return history

    def train_epoch(self, x_train: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in self._batch_iterator(
            x_train, y_train, batch_size=self.config.batch_size, shuffle=self.config.shuffle
        ):
            logits = self.model.forward(x_batch, training=True)
            loss_value = self.loss_fn.forward(logits, y_batch)
            grad_logits = self.loss_fn.backward(logits, y_batch)
            _ = self.model.backward(grad_logits)
            self.optimizer.step(self.model)

            total_loss += loss_value
            num_batches += 1

        mean_loss = total_loss / max(num_batches, 1)
        return {"loss": float(mean_loss)}

    def evaluate(self, x_data: np.ndarray, y_data: np.ndarray) -> dict[str, Any]:
        logits = self.model.forward(x_data, training=False)
        accuracy = self._accuracy_from_logits(logits, y_data)
        return {"accuracy": float(accuracy)}

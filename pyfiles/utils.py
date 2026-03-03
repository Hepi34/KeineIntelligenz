"""Training and evaluation helper utilities."""

from __future__ import annotations

import time
from collections.abc import Generator

import numpy as np


def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute classification accuracy."""
    raise NotImplementedError


def batch_iterator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield mini-batches from arrays."""
    raise NotImplementedError


class Timer:
    """Simple wall-clock timer utility."""

    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()
        self._end = None

    def stop(self) -> float:
        self._end = time.perf_counter()
        if self._start is None:
            raise RuntimeError("Timer was not started.")
        return self._end - self._start

    @property
    def elapsed(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer was not started.")
        if self._end is not None:
            return self._end - self._start
        return time.perf_counter() - self._start


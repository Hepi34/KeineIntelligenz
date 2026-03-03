"""Optimizer implementations."""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np


class Parameterized(Protocol):
    """Protocol for models/layers exposing trainable params and grads."""

    def parameters(self) -> list[np.ndarray]:
        ...

    def gradients(self) -> list[np.ndarray]:
        ...


class Optimizer:
    """Base optimizer class."""

    def step(
        self,
        model_or_params: Parameterized | Iterable[np.ndarray],
        grads: Iterable[np.ndarray] | None = None,
    ) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        """Optional API for compatibility; no-op by default."""
        return None


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def __init__(self, lr: float = 0.01) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be > 0, got {lr}.")
        self.lr = lr

    def step(
        self,
        model_or_params: Parameterized | Iterable[np.ndarray],
        grads: Iterable[np.ndarray] | None = None,
    ) -> None:
        """
        Update parameters with SGD.

        Supported usage:
        - step(model) where model exposes parameters() and gradients()
        - step(params, grads)
        """
        if grads is None:
            if not hasattr(model_or_params, "parameters") or not hasattr(model_or_params, "gradients"):
                raise ValueError("When grads is None, first argument must expose parameters() and gradients().")
            params_iter = model_or_params.parameters()  # type: ignore[union-attr]
            grads_iter = model_or_params.gradients()  # type: ignore[union-attr]
        else:
            params_iter = model_or_params  # type: ignore[assignment]
            grads_iter = grads

        params = list(params_iter)
        grad_list = list(grads_iter)
        if len(params) != len(grad_list):
            raise ValueError(
                f"Parameter/gradient length mismatch: {len(params)} params vs {len(grad_list)} grads."
            )

        for param, grad in zip(params, grad_list, strict=True):
            if param.shape != grad.shape:
                raise ValueError(
                    f"Parameter/gradient shape mismatch: param {param.shape}, grad {grad.shape}."
                )
            param -= self.lr * grad

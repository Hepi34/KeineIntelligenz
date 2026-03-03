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


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be > 0, got {lr}.")
        if not (0.0 < beta1 < 1.0):
            raise ValueError(f"beta1 must be in (0, 1), got {beta1}.")
        if not (0.0 < beta2 < 1.0):
            raise ValueError(f"beta2 must be in (0, 1), got {beta2}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}

    def step(
        self,
        model_or_params: Parameterized | Iterable[np.ndarray],
        grads: Iterable[np.ndarray] | None = None,
    ) -> None:
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

        self.t += 1
        beta1_t = self.beta1 ** self.t
        beta2_t = self.beta2 ** self.t

        for param, grad in zip(params, grad_list, strict=True):
            if param.shape != grad.shape:
                raise ValueError(
                    f"Parameter/gradient shape mismatch: param {param.shape}, grad {grad.shape}."
                )
            key = id(param)
            if key not in self.m:
                self.m[key] = np.zeros_like(param, dtype=np.float32)
                self.v[key] = np.zeros_like(param, dtype=np.float32)

            g = grad.astype(np.float32, copy=False)
            m = self.m[key]
            v = self.v[key]
            m[...] = self.beta1 * m + (1.0 - self.beta1) * g
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            m_hat = m / (1.0 - beta1_t)
            v_hat = v / (1.0 - beta2_t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

"""Core neural network layer definitions."""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class for all layers."""

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> list[np.ndarray]:
        return []

    def gradients(self) -> list[np.ndarray]:
        return []


class Conv(Layer):
    """Compatibility alias for Conv2D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        self._impl = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return self._impl.forward(x, training=training)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return self._impl.backward(grad_output)

    def parameters(self) -> list[np.ndarray]:
        return self._impl.parameters()

    def gradients(self) -> list[np.ndarray]:
        return self._impl.gradients()


class Conv2D(Layer):
    """2D convolution layer for NCHW tensors with 3x3 kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        if kernel_size != 3:
            raise ValueError(f"Conv2D supports only 3x3 kernels, got {kernel_size}.")
        if stride <= 0:
            raise ValueError(f"Stride must be >= 1, got {stride}.")
        if padding < 0:
            raise ValueError(f"Padding must be >= 0, got {padding}.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialization for ReLU-based conv stacks.
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.weights = (
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        )
        self.bias = np.zeros((out_channels,), dtype=np.float32)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self._x: np.ndarray | None = None
        self._x_padded: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects input shape (N, C, H, W), got ndim={x.ndim}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channels mismatch: layer expects {self.in_channels}, got {x.shape[1]}."
            )

        x = x.astype(np.float32, copy=False)
        n, _, h, w = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        out_h = (h + 2 * p - k) // s + 1
        out_w = (w + 2 * p - k) // s + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                f"Invalid output shape with input {(h, w)}, kernel {k}, stride {s}, padding {p}."
            )

        if p > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
        else:
            x_padded = x

        out = np.zeros((n, self.out_channels, out_h, out_w), dtype=np.float32)
        for i in range(out_h):
            hs = i * s
            he = hs + k
            for j in range(out_w):
                ws = j * s
                we = ws + k
                region = x_padded[:, :, hs:he, ws:we]  # (N, Cin, k, k)
                # Sum over Cin, k, k for each sample and filter.
                out[:, :, i, j] = np.tensordot(region, self.weights, axes=([1, 2, 3], [1, 2, 3])) + self.bias

        self._x = x
        self._x_padded = x_padded
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x is None or self._x_padded is None:
            raise RuntimeError("Call forward before backward.")

        grad_output = grad_output.astype(np.float32, copy=False)
        x = self._x
        x_padded = self._x_padded

        n, _, h, w = x.shape
        _, _, out_h, out_w = grad_output.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        self.grad_weights.fill(0.0)
        self.grad_bias[...] = grad_output.sum(axis=(0, 2, 3))

        grad_x_padded = np.zeros_like(x_padded, dtype=np.float32)
        for i in range(out_h):
            hs = i * s
            he = hs + k
            for j in range(out_w):
                ws = j * s
                we = ws + k
                region = x_padded[:, :, hs:he, ws:we]  # (N, Cin, k, k)
                go = grad_output[:, :, i, j]  # (N, Cout)

                # dW[c_out] += sum_n dY[n, c_out] * X_patch[n]
                self.grad_weights += np.tensordot(go, region, axes=([0], [0]))

                # dX_patch[n] += sum_c_out dY[n, c_out] * W[c_out]
                grad_x_padded[:, :, hs:he, ws:we] += np.tensordot(go, self.weights, axes=([1], [0]))

        if p > 0:
            grad_x = grad_x_padded[:, :, p : p + h, p : p + w]
        else:
            grad_x = grad_x_padded
        return grad_x

    def parameters(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    def gradients(self) -> list[np.ndarray]:
        return [self.grad_weights, self.grad_bias]


class Dense(Layer):
    """Fully connected layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(2.0 / in_features)
        self.weights = (np.random.randn(in_features, out_features).astype(np.float32) * scale)
        self.bias = np.zeros((out_features,), dtype=np.float32)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Dense expects input shape (N, F), got {x.shape}.")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Input features mismatch: layer expects {self.in_features}, got {x.shape[1]}."
            )
        x = x.astype(np.float32, copy=False)
        self._x = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("Call forward before backward.")
        grad_output = grad_output.astype(np.float32, copy=False)

        self.grad_weights[...] = self._x.T @ grad_output
        self.grad_bias[...] = grad_output.sum(axis=0)
        grad_input = grad_output @ self.weights.T
        return grad_input

    def parameters(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    def gradients(self) -> list[np.ndarray]:
        return [self.grad_weights, self.grad_bias]


class ReLU(Layer):
    """ReLU activation layer."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._mask = x > 0
        return np.maximum(x, 0).astype(np.float32, copy=False)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("Call forward before backward.")
        return grad_output * self._mask


class MaxPool2D(Layer):
    """2D max pooling for NCHW tensors."""

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}.")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}.")
        self.kernel_size = kernel_size
        self.stride = stride
        self._x_shape: tuple[int, ...] | None = None
        self._argmax: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"MaxPool2D expects input shape (N, C, H, W), got {x.shape}.")
        x = x.astype(np.float32, copy=False)
        n, c, h, w = x.shape
        k = self.kernel_size
        s = self.stride

        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Invalid MaxPool2D output shape from input {x.shape} with k={k}, s={s}.")

        out = np.empty((n, c, out_h, out_w), dtype=np.float32)
        argmax = np.empty((n, c, out_h, out_w), dtype=np.int32)

        for oh in range(out_h):
            hs = oh * s
            he = hs + k
            for ow in range(out_w):
                ws = ow * s
                we = ws + k
                window = x[:, :, hs:he, ws:we]  # (N, C, k, k)
                flat = window.reshape(n, c, -1)
                idx = np.argmax(flat, axis=2).astype(np.int32)
                out[:, :, oh, ow] = np.take_along_axis(flat, idx[:, :, None], axis=2).squeeze(axis=2)
                argmax[:, :, oh, ow] = idx

        self._x_shape = x.shape
        self._argmax = argmax
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._x_shape is None or self._argmax is None:
            raise RuntimeError("Call forward before backward.")
        grad_output = grad_output.astype(np.float32, copy=False)
        n, c, h, w = self._x_shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        k = self.kernel_size
        s = self.stride

        grad_input = np.zeros((n, c, h, w), dtype=np.float32)
        for oh in range(out_h):
            hs = oh * s
            for ow in range(out_w):
                ws = ow * s
                idx = self._argmax[:, :, oh, ow]
                idx_h = idx // k
                idx_w = idx % k
                n_idx = np.arange(n)[:, None]
                c_idx = np.arange(c)[None, :]
                grad_input[n_idx, c_idx, hs + idx_h, ws + idx_w] += grad_output[:, :, oh, ow]
        return grad_input


class Softmax(Layer):
    """Softmax activation over class dimension."""

    def __init__(self) -> None:
        self._output: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim != 2:
            raise ValueError(f"Softmax expects input shape (N, C), got {x.shape}.")
        x = x.astype(np.float32, copy=False)
        shifted = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self._output = probs
        return probs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._output is None:
            raise RuntimeError("Call forward before backward.")
        # Vectorized Jacobian-vector product for each sample:
        # dL/dz = s * (dL/ds - sum(dL/ds * s))
        s = self._output
        dot = np.sum(grad_output * s, axis=1, keepdims=True)
        return s * (grad_output - dot)


class Flatten(Layer):
    """Flatten N-D input to 2D (batch, features)."""

    def __init__(self) -> None:
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if x.ndim < 2:
            raise ValueError(f"Flatten expects at least 2D input, got {x.shape}.")
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("Call forward before backward.")
        return grad_output.reshape(self._input_shape)

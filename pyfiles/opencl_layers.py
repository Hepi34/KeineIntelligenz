# Hepi34, onatic07, fritziii, 2026-03-06

"""OpenCL-backed layer implementations."""

from __future__ import annotations

import numpy as np

from opencl_backend import OpenCLManager, cl


_CONV2D_FORWARD_KERNEL = r"""
__kernel void conv2d_forward_nchw(
    __global const float* x,        // [N, Cin, H, W]
    __global const float* w,        // [Cout, Cin, K, K]
    __global const float* b,        // [Cout]
    __global float* out,            // [N, Cout, OH, OW]
    const int N,
    const int Cin,
    const int H,
    const int W,
    const int Cout,
    const int K,
    const int stride,
    const int padding,
    const int OH,
    const int OW
) {
    const int n = get_global_id(0);
    const int co = get_global_id(1);
    const int flat = get_global_id(2);

    if (n >= N || co >= Cout || flat >= OH * OW) {
        return;
    }

    const int oh = flat / OW;
    const int ow = flat % OW;
    const int h0 = oh * stride - padding;
    const int w0 = ow * stride - padding;

    float acc = b[co];

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int ih = h0 + kh;
                const int iw = w0 + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    const int x_idx = ((n * Cin + ci) * H + ih) * W + iw;
                    const int w_idx = ((co * Cin + ci) * K + kh) * K + kw;
                    acc += x[x_idx] * w[w_idx];
                }
            }
        }
    }

    const int out_idx = ((n * Cout + co) * OH + oh) * OW + ow;
    out[out_idx] = acc;
}
"""


class OpenCLConv2D:
    """
    OpenCL Conv2D forward pass for NCHW tensors.

    Forward convolution is executed entirely on GPU via OpenCL kernel.
    """

    def __init__(
        self,
        opencl: OpenCLManager,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        if cl is None:
            raise RuntimeError("PyOpenCL is not available.")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}.")
        if stride <= 0:
            raise ValueError(f"stride must be > 0, got {stride}.")
        if padding < 0:
            raise ValueError(f"padding must be >= 0, got {padding}.")

        self.opencl = opencl
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.weights = (
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        )
        self.bias = np.zeros((out_channels,), dtype=np.float32)

        self.w_buffer = self.opencl.to_device(self.weights)
        self.b_buffer = self.opencl.to_device(self.bias)

        self._program = cl.Program(self.opencl.context, _CONV2D_FORWARD_KERNEL).build()

    def set_weights(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """Replace weights/bias and upload to GPU buffers."""
        if weights.shape != self.weights.shape:
            raise ValueError(f"weights shape mismatch: expected {self.weights.shape}, got {weights.shape}.")
        if bias.shape != self.bias.shape:
            raise ValueError(f"bias shape mismatch: expected {self.bias.shape}, got {bias.shape}.")
        self.weights[...] = weights.astype(np.float32, copy=False)
        self.bias[...] = bias.astype(np.float32, copy=False)
        self.w_buffer = self.opencl.to_device(self.weights)
        self.b_buffer = self.opencl.to_device(self.bias)

    def output_shape(self, h: int, w: int) -> tuple[int, int]:
        oh = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        ow = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        if oh <= 0 or ow <= 0:
            raise ValueError(
                f"Invalid output shape for input {(h, w)} and config "
                f"(K={self.kernel_size}, stride={self.stride}, padding={self.padding})."
            )
        return oh, ow

    def forward_device(self, x_buffer: object, n: int, h: int, w: int) -> tuple[object, tuple[int, int, int, int]]:
        """
        Run Conv2D forward pass on GPU buffers.

        Args:
            x_buffer: OpenCL buffer for input tensor [N, Cin, H, W], float32.
            n: Batch size N.
            h: Input height H.
            w: Input width W.

        Returns:
            (out_buffer, out_shape) where out_shape is (N, Cout, OH, OW).
        """
        if cl is None:
            raise RuntimeError("PyOpenCL is not available.")

        oh, ow = self.output_shape(h, w)
        out_shape = (n, self.out_channels, oh, ow)
        out_buffer = self.opencl.empty_device(out_shape, np.dtype(np.float32))

        global_size = (int(n), int(self.out_channels), int(oh * ow))
        self._program.conv2d_forward_nchw(
            self.opencl.queue,
            global_size,
            None,
            x_buffer,
            self.w_buffer,
            self.b_buffer,
            out_buffer,
            np.int32(n),
            np.int32(self.in_channels),
            np.int32(h),
            np.int32(w),
            np.int32(self.out_channels),
            np.int32(self.kernel_size),
            np.int32(self.stride),
            np.int32(self.padding),
            np.int32(oh),
            np.int32(ow),
        )
        self.opencl.queue.finish()
        return out_buffer, out_shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Host convenience wrapper:
        upload input -> run OpenCL kernel -> download output.
        Convolution computation itself is GPU-only (no CPU fallback).
        """
        if x.ndim != 4:
            raise ValueError(f"OpenCLConv2D expects input shape (N, C, H, W), got {x.shape}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Input channels mismatch: expected {self.in_channels}, got {x.shape[1]}.")

        x_host = np.ascontiguousarray(x.astype(np.float32, copy=False))
        x_buffer = self.opencl.to_device(x_host)
        out_buffer, out_shape = self.forward_device(x_buffer, n=x_host.shape[0], h=x_host.shape[2], w=x_host.shape[3])
        return self.opencl.from_device(out_buffer, out_shape, np.dtype(np.float32))

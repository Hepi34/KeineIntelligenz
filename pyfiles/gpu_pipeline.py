"""End-to-end OpenCL training pipeline for MNIST CNN (GPU-only math path)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from layers import Conv2D, Dense, Flatten, ReLU
from model import CNNModel
from opencl_backend import OpenCLManager, cl


KERNEL_SOURCE = r"""
__kernel void copy_batch_f32(
    __global const float* src,
    __global float* dst,
    const int start_sample,
    const int sample_elems,
    const int n_samples
) {
    const int gid = get_global_id(0);
    const int total = n_samples * sample_elems;
    if (gid >= total) return;
    dst[gid] = src[start_sample * sample_elems + gid];
}

__kernel void copy_batch_i32(
    __global const int* src,
    __global int* dst,
    const int start_sample,
    const int n_samples
) {
    const int gid = get_global_id(0);
    if (gid >= n_samples) return;
    dst[gid] = src[start_sample + gid];
}

__kernel void conv2d_forward_nchw(
    __global const float* x,
    __global const float* w,
    __global const float* b,
    __global float* out,
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
    // Launch-shape sanity: host is expected to launch exactly (N, Cout, OH*OW).
    // Keep bounds checks regardless to avoid OOB access with mismatched global sizes.
    if ((int)get_global_size(0) < N || (int)get_global_size(1) < Cout || (int)get_global_size(2) < OH * OW) return;
    if (N <= 0 || Cin <= 0 || H <= 0 || W <= 0 || Cout <= 0 || K <= 0 || OH <= 0 || OW <= 0) return;
    if (n >= N || co >= Cout || flat >= OH * OW) return;

    const int oh = flat / OW;
    const int ow = flat % OW;
    const int h0 = oh * stride - padding;
    const int w0 = ow * stride - padding;

    const int x_total = N * Cin * H * W;
    const int w_total = Cout * Cin * K * K;
    const int out_total = N * Cout * OH * OW;

    // Start from zeroed accumulator; each output is computed from scratch.
    float value = 0.0f;
    if (co >= 0 && co < Cout) {
        value += b[co];
    }

    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int ih = h0 + kh;
                const int iw = w0 + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    // NCHW linear index: n*C*H*W + c*H*W + h*W + w
                    const int x_idx = n * (Cin * H * W) + ci * (H * W) + ih * W + iw;
                    const int w_idx = ((co * Cin + ci) * K + kh) * K + kw;
                    if (x_idx >= 0 && x_idx < x_total && w_idx >= 0 && w_idx < w_total) {
                        value += x[x_idx] * w[w_idx];
                    }
                }
            }
        }
    }

    if (!isfinite(value)) value = 0.0f;
    if (value > 1e6f) value = 1e6f;
    if (value < -1e6f) value = -1e6f;

    const int out_idx = ((n * Cout + co) * OH + oh) * OW + ow;
    if (out_idx >= 0 && out_idx < out_total) {
        out[out_idx] = value;
    }
}

__kernel void conv2d_backward_weight_nchw(
    __global const float* x,
    __global const float* grad_out,
    __global float* grad_w,
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
    const int co = get_global_id(0);
    const int ci = get_global_id(1);
    const int kflat = get_global_id(2);
    if (co >= Cout || ci >= Cin || kflat >= K * K) return;

    const int kh = kflat / K;
    const int kw = kflat % K;
    float acc = 0.0f;

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                const int ih = oh * stride - padding + kh;
                const int iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    const int x_idx = ((n * Cin + ci) * H + ih) * W + iw;
                    const int go_idx = ((n * Cout + co) * OH + oh) * OW + ow;
                    acc += x[x_idx] * grad_out[go_idx];
                }
            }
        }
    }

    const int gw_idx = ((co * Cin + ci) * K + kh) * K + kw;
    grad_w[gw_idx] = acc;
}

__kernel void conv2d_backward_bias_nchw(
    __global const float* grad_out,
    __global float* grad_b,
    const int N,
    const int Cout,
    const int OH,
    const int OW
) {
    const int co = get_global_id(0);
    if (co >= Cout) return;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                const int idx = ((n * Cout + co) * OH + oh) * OW + ow;
                acc += grad_out[idx];
            }
        }
    }
    grad_b[co] = acc;
}

__kernel void conv2d_backward_input_nchw(
    __global const float* grad_out,
    __global const float* w,
    __global float* grad_x,
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
    const int ci = get_global_id(1);
    const int hw = get_global_id(2);
    if (n >= N || ci >= Cin || hw >= H * W) return;

    const int ih = hw / W;
    const int iw = hw % W;
    float acc = 0.0f;

    for (int co = 0; co < Cout; ++co) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                const int oh_num = ih + padding - kh;
                const int ow_num = iw + padding - kw;
                if (oh_num % stride != 0 || ow_num % stride != 0) continue;
                const int oh = oh_num / stride;
                const int ow = ow_num / stride;
                if (oh >= 0 && oh < OH && ow >= 0 && ow < OW) {
                    const int go_idx = ((n * Cout + co) * OH + oh) * OW + ow;
                    const int w_idx = ((co * Cin + ci) * K + kh) * K + kw;
                    acc += grad_out[go_idx] * w[w_idx];
                }
            }
        }
    }

    const int gx_idx = ((n * Cin + ci) * H + ih) * W + iw;
    grad_x[gx_idx] = acc;
}

__kernel void relu_forward(
    __global const float* x,
    __global float* y,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    float v = x[gid];
    if (!isfinite(v)) v = 0.0f;
    y[gid] = v > 0.0f ? v : 0.0f;
}

__kernel void relu_backward(
    __global const float* grad_out,
    __global const float* x,
    __global float* grad_x,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    grad_x[gid] = x[gid] > 0.0f ? grad_out[gid] : 0.0f;
}

__kernel void dense_forward(
    __global const float* x,
    __global const float* w,
    __global const float* b,
    __global float* out,
    const int N,
    const int IN,
    const int OUT
) {
    const int n = get_global_id(0);
    const int o = get_global_id(1);
    if (n >= N || o >= OUT) return;

    float acc = b[o];
    for (int i = 0; i < IN; ++i) {
        acc += x[n * IN + i] * w[i * OUT + o];
    }
    if (!isfinite(acc)) acc = 0.0f;
    if (acc > 1e6f) acc = 1e6f;
    if (acc < -1e6f) acc = -1e6f;
    out[n * OUT + o] = acc;
}

__kernel void dense_backward_input(
    __global const float* grad_out,
    __global const float* w,
    __global float* grad_x,
    const int N,
    const int IN,
    const int OUT
) {
    const int n = get_global_id(0);
    const int i = get_global_id(1);
    if (n >= N || i >= IN) return;

    float acc = 0.0f;
    for (int o = 0; o < OUT; ++o) {
        acc += grad_out[n * OUT + o] * w[i * OUT + o];
    }
    grad_x[n * IN + i] = acc;
}

__kernel void dense_backward_weight(
    __global const float* x,
    __global const float* grad_out,
    __global float* grad_w,
    const int N,
    const int IN,
    const int OUT
) {
    const int i = get_global_id(0);
    const int o = get_global_id(1);
    if (i >= IN || o >= OUT) return;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        acc += x[n * IN + i] * grad_out[n * OUT + o];
    }
    grad_w[i * OUT + o] = acc;
}

__kernel void dense_backward_bias(
    __global const float* grad_out,
    __global float* grad_b,
    const int N,
    const int OUT
) {
    const int o = get_global_id(0);
    if (o >= OUT) return;
    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        acc += grad_out[n * OUT + o];
    }
    grad_b[o] = acc;
}

__kernel void softmax_xent_grad(
    __global const float* logits,
    __global const int* labels,
    __global float* grad_logits,
    __global float* loss_per_sample,
    const int N,
    const int C
) {
    const int n = get_global_id(0);
    if (n >= N) return;

    const int row = n * C;
    float first = logits[row];
    if (!isfinite(first)) first = 0.0f;
    float max_v = first;
    for (int c = 1; c < C; ++c) {
        float v = logits[row + c];
        if (!isfinite(v)) v = 0.0f;
        if (v > max_v) max_v = v;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        float v = logits[row + c];
        if (!isfinite(v)) v = 0.0f;
        sum_exp += exp(v - max_v);
    }
    if (!isfinite(sum_exp) || sum_exp <= 0.0f) sum_exp = 1.0f;

    const int y = labels[n];
    float p_y = 0.0f;
    for (int c = 0; c < C; ++c) {
        float v = logits[row + c];
        if (!isfinite(v)) v = 0.0f;
        float p = exp(v - max_v) / sum_exp;
        if (!isfinite(p)) p = 0.0f;
        if (c == y) p_y = p;
        const float t = (c == y) ? 1.0f : 0.0f;
        grad_logits[row + c] = (p - t) / (float)N;
    }

    if (p_y < 1e-12f) p_y = 1e-12f;
    loss_per_sample[n] = -log(p_y);
}

__kernel void reduce_sum_f32_single(
    __global const float* x,
    __global float* out,
    const int n
) {
    if (get_global_id(0) != 0) return;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += x[i];
    out[0] = acc;
}

__kernel void set_f32_zero(__global float* x) {
    if (get_global_id(0) != 0) return;
    x[0] = 0.0f;
}

__kernel void add_f32_scalar(
    __global float* acc,
    __global const float* x
) {
    if (get_global_id(0) != 0) return;
    acc[0] += x[0];
}

__kernel void div_f32_by_i32(
    __global const float* x,
    __global float* out,
    const int denom
) {
    if (get_global_id(0) != 0) return;
    out[0] = x[0] / (float)denom;
}

__kernel void accuracy_flag_from_logits(
    __global const float* logits,
    __global const int* labels,
    __global float* flags,
    const int N,
    const int C
) {
    const int n = get_global_id(0);
    if (n >= N) return;
    const int row = n * C;
    int argmax = 0;
    float max_v = logits[row];
    for (int c = 1; c < C; ++c) {
        const float v = logits[row + c];
        if (v > max_v) {
            max_v = v;
            argmax = c;
        }
    }
    flags[n] = (argmax == labels[n]) ? 1.0f : 0.0f;
}

__kernel void sgd_update(
    __global float* param,
    __global const float* grad,
    const float lr,
    const int n
) {
    const int gid = get_global_id(0);
    if (gid >= n) return;
    float p = param[gid];
    float g = grad[gid];
    if (!isfinite(p)) p = 0.0f;
    if (!isfinite(g)) g = 0.0f;
    // Clip gradient to prevent explosion
    const float grad_clip = 10.0f;
    if (g > grad_clip) g = grad_clip;
    if (g < -grad_clip) g = -grad_clip;
    p -= lr * g;
    // Also clip parameters to prevent extreme values
    if (p > 1e6f) p = 1e6f;
    if (p < -1e6f) p = -1e6f;
    param[gid] = p;
}
"""


@dataclass(frozen=True)
class GPUTrainConfig:
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.01
    conv_filters: int = 8
    hidden_units: int = 64
    num_classes: int = 10
    input_channels: int = 1
    input_height: int = 28
    input_width: int = 28
    kernel_size: int = 3
    stride: int = 1
    padding: int = 0
    shuffle: bool = True
    debug_mode: bool = False
    debug_batch_size: int = 8
    debug_rtol: float = 1e-4
    debug_atol: float = 1e-5


class GPUTrainingPipeline:
    """OpenCL-only training path for Conv->ReLU->Dense->ReLU->Dense->Softmax."""

    def __init__(self, manager: OpenCLManager, config: GPUTrainConfig) -> None:
        if cl is None:
            raise RuntimeError("PyOpenCL is not available.")
        self.manager = manager
        self.cfg = config
        self.program = cl.Program(self.manager.context, KERNEL_SOURCE).build()
        self._init_kernels()

        self.oh = (config.input_height + 2 * config.padding - config.kernel_size) // config.stride + 1
        self.ow = (config.input_width + 2 * config.padding - config.kernel_size) // config.stride + 1
        if self.oh <= 0 or self.ow <= 0:
            raise ValueError("Invalid conv output shape; check kernel/stride/padding.")

        self.flat_features = config.conv_filters * self.oh * self.ow
        self._init_parameters()

    def _init_kernels(self) -> None:
        if cl is None:
            raise RuntimeError("PyOpenCL is not available.")

        # Cache kernel objects once to avoid repeated retrieval overhead/warnings.
        self.k_copy_batch_f32 = cl.Kernel(self.program, "copy_batch_f32")
        self.k_copy_batch_i32 = cl.Kernel(self.program, "copy_batch_i32")
        self.k_conv2d_forward_nchw = cl.Kernel(self.program, "conv2d_forward_nchw")
        self.k_conv2d_backward_weight_nchw = cl.Kernel(self.program, "conv2d_backward_weight_nchw")
        self.k_conv2d_backward_bias_nchw = cl.Kernel(self.program, "conv2d_backward_bias_nchw")
        self.k_conv2d_backward_input_nchw = cl.Kernel(self.program, "conv2d_backward_input_nchw")
        self.k_relu_forward = cl.Kernel(self.program, "relu_forward")
        self.k_relu_backward = cl.Kernel(self.program, "relu_backward")
        self.k_dense_forward = cl.Kernel(self.program, "dense_forward")
        self.k_dense_backward_input = cl.Kernel(self.program, "dense_backward_input")
        self.k_dense_backward_weight = cl.Kernel(self.program, "dense_backward_weight")
        self.k_dense_backward_bias = cl.Kernel(self.program, "dense_backward_bias")
        self.k_softmax_xent_grad = cl.Kernel(self.program, "softmax_xent_grad")
        self.k_reduce_sum_f32_single = cl.Kernel(self.program, "reduce_sum_f32_single")
        self.k_set_f32_zero = cl.Kernel(self.program, "set_f32_zero")
        self.k_add_f32_scalar = cl.Kernel(self.program, "add_f32_scalar")
        self.k_div_f32_by_i32 = cl.Kernel(self.program, "div_f32_by_i32")
        self.k_accuracy_flag_from_logits = cl.Kernel(self.program, "accuracy_flag_from_logits")
        self.k_sgd_update = cl.Kernel(self.program, "sgd_update")

    def _init_parameters(self) -> None:
        cfg = self.cfg
        he_conv = np.float32(np.sqrt(np.float32(2.0) / np.float32(cfg.input_channels * cfg.kernel_size * cfg.kernel_size)))
        he_fc1 = np.float32(np.sqrt(np.float32(2.0) / np.float32(self.flat_features)))
        he_fc2 = np.float32(np.sqrt(np.float32(2.0) / np.float32(cfg.hidden_units)))

        # Create CPU arrays and KEEP REFERENCES to them
        self._cpu_w_conv = (
            np.random.randn(cfg.conv_filters, cfg.input_channels, cfg.kernel_size, cfg.kernel_size).astype(np.float32)
            * he_conv
        ).astype(np.float32, copy=False)
        self._cpu_b_conv = np.zeros((cfg.conv_filters,), dtype=np.float32)
        self._cpu_w_fc1 = (
            np.random.randn(self.flat_features, cfg.hidden_units).astype(np.float32) * he_fc1
        ).astype(np.float32, copy=False)
        self._cpu_b_fc1 = np.zeros((cfg.hidden_units,), dtype=np.float32)
        self._cpu_w_fc2 = (
            np.random.randn(cfg.hidden_units, cfg.num_classes).astype(np.float32) * he_fc2
        ).astype(np.float32, copy=False)
        self._cpu_b_fc2 = np.zeros((cfg.num_classes,), dtype=np.float32)

        # Use the safer to_device method for all uploads
        self.w_conv = self.manager.to_device(self._cpu_w_conv)
        self.b_conv = self.manager.to_device(self._cpu_b_conv)
        self.w_fc1 = self.manager.to_device(self._cpu_w_fc1)
        self.b_fc1 = self.manager.to_device(self._cpu_b_fc1)
        self.w_fc2 = self.manager.to_device(self._cpu_w_fc2)
        self.b_fc2 = self.manager.to_device(self._cpu_b_fc2)

        # Allocate gradient buffers AFTER all params are uploaded
        self.gw_conv = self.manager.empty_device(self._cpu_w_conv.shape, np.dtype(np.float32))
        self.gb_conv = self.manager.empty_device(self._cpu_b_conv.shape, np.dtype(np.float32))
        self.gw_fc1 = self.manager.empty_device(self._cpu_w_fc1.shape, np.dtype(np.float32))
        self.gb_fc1 = self.manager.empty_device(self._cpu_b_fc1.shape, np.dtype(np.float32))
        self.gw_fc2 = self.manager.empty_device(self._cpu_w_fc2.shape, np.dtype(np.float32))
        self.gb_fc2 = self.manager.empty_device(self._cpu_b_fc2.shape, np.dtype(np.float32))

    def _alloc_batch_buffers(self, batch_size: int) -> dict[str, Any]:
        cfg = self.cfg
        return {
            "x": self.manager.empty_device(
                (batch_size, cfg.input_channels, cfg.input_height, cfg.input_width), np.dtype(np.float32)
            ),
            "y": self.manager.empty_device((batch_size,), np.dtype(np.int32)),
            "z_conv": self.manager.empty_device((batch_size, cfg.conv_filters, self.oh, self.ow), np.dtype(np.float32)),
            "a_conv": self.manager.empty_device((batch_size, cfg.conv_filters, self.oh, self.ow), np.dtype(np.float32)),
            "z_fc1": self.manager.empty_device((batch_size, cfg.hidden_units), np.dtype(np.float32)),
            "a_fc1": self.manager.empty_device((batch_size, cfg.hidden_units), np.dtype(np.float32)),
            "logits": self.manager.empty_device((batch_size, cfg.num_classes), np.dtype(np.float32)),
            "d_logits": self.manager.empty_device((batch_size, cfg.num_classes), np.dtype(np.float32)),
            "d_a_fc1": self.manager.empty_device((batch_size, cfg.hidden_units), np.dtype(np.float32)),
            "d_z_fc1": self.manager.empty_device((batch_size, cfg.hidden_units), np.dtype(np.float32)),
            "d_a_conv": self.manager.empty_device((batch_size, self.flat_features), np.dtype(np.float32)),
            "d_z_conv": self.manager.empty_device((batch_size, cfg.conv_filters, self.oh, self.ow), np.dtype(np.float32)),
            "d_x": self.manager.empty_device(
                (batch_size, cfg.input_channels, cfg.input_height, cfg.input_width), np.dtype(np.float32)
            ),
            "losses": self.manager.empty_device((batch_size,), np.dtype(np.float32)),
            "loss_sum": self.manager.empty_device((1,), np.dtype(np.float32)),
            "loss_accum": self.manager.empty_device((1,), np.dtype(np.float32)),
            "metric_f32": self.manager.empty_device((1,), np.dtype(np.float32)),
            "correct_flags": self.manager.empty_device((batch_size,), np.dtype(np.float32)),
            "correct_sum": self.manager.empty_device((1,), np.dtype(np.float32)),
            "correct_accum": self.manager.empty_device((1,), np.dtype(np.float32)),
        }

    def _parameter_snapshot(self) -> dict[str, np.ndarray]:
        cfg = self.cfg
        return {
            "w_conv": self.manager.from_device(
                self.w_conv,
                (cfg.conv_filters, cfg.input_channels, cfg.kernel_size, cfg.kernel_size),
                np.dtype(np.float32),
            ),
            "b_conv": self.manager.from_device(self.b_conv, (cfg.conv_filters,), np.dtype(np.float32)),
            "w_fc1": self.manager.from_device(
                self.w_fc1,
                (self.flat_features, cfg.hidden_units),
                np.dtype(np.float32),
            ),
            "b_fc1": self.manager.from_device(self.b_fc1, (cfg.hidden_units,), np.dtype(np.float32)),
            "w_fc2": self.manager.from_device(
                self.w_fc2,
                (cfg.hidden_units, cfg.num_classes),
                np.dtype(np.float32),
            ),
            "b_fc2": self.manager.from_device(self.b_fc2, (cfg.num_classes,), np.dtype(np.float32)),
        }

    @staticmethod
    def _finite_stats(name: str, arr: np.ndarray) -> str:
        finite = np.isfinite(arr)
        return (
            f"{name}: finite={bool(finite.all())} "
            f"min={float(np.nanmin(arr)):.6f} max={float(np.nanmax(arr)):.6f} "
            f"mean={float(np.nanmean(arr)):.6f} std={float(np.nanstd(arr)):.6f}"
        )

    @staticmethod
    def _stable_softmax(logits: np.ndarray) -> np.ndarray:
        z = logits.astype(np.float32, copy=False)
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        denom = np.sum(exp_shifted, axis=1, keepdims=True)
        denom = np.maximum(denom, np.float32(1e-12))
        return exp_shifted / denom

    @staticmethod
    def _tensor_stats(arr: np.ndarray) -> dict[str, float | list[float]]:
        flat = arr.reshape(-1).astype(np.float64, copy=False)
        first = flat[:5]
        return {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "first5": [float(v) for v in first],
        }

    @staticmethod
    def _format_stats(stats: dict[str, float | list[float]]) -> str:
        first5 = stats["first5"]
        assert isinstance(first5, list)
        first5_str = ", ".join(f"{float(v):.6g}" for v in first5)
        return (
            f"min={float(stats['min']):.6g} "
            f"max={float(stats['max']):.6g} "
            f"mean={float(stats['mean']):.6g} "
            f"std={float(stats['std']):.6g} "
            f"first5=[{first5_str}]"
        )

    @staticmethod
    def _exp_overflow_flags(logits: np.ndarray) -> tuple[bool, bool, float]:
        z = logits.astype(np.float32, copy=False)
        max_logit = float(np.max(z))
        overflow_threshold = float(np.log(np.finfo(np.float32).max))
        would_overflow = bool(np.any(z > overflow_threshold))
        with np.errstate(over="ignore", invalid="ignore"):
            naive_exp = np.exp(z)
        has_inf = bool(np.isinf(naive_exp).any())
        return would_overflow, has_inf, max_logit

    def _cpu_forward_intermediates(self, x: np.ndarray) -> dict[str, np.ndarray]:
        cpu_model = self.to_cpu_model()
        conv = cpu_model.layers[0]
        relu1 = cpu_model.layers[1]
        flatten = cpu_model.layers[2]
        fc1 = cpu_model.layers[3]
        relu2 = cpu_model.layers[4]
        fc2 = cpu_model.layers[5]

        z_conv = conv.forward(x, training=False)
        a_conv = relu1.forward(z_conv, training=False)
        flat = flatten.forward(a_conv, training=False)
        z_fc1 = fc1.forward(flat, training=False)
        a_fc1 = relu2.forward(z_fc1, training=False)
        logits = fc2.forward(a_fc1, training=False)
        probs = self._stable_softmax(logits)
        return {
            "z_conv": z_conv.astype(np.float32, copy=False),
            "a_conv": a_conv.astype(np.float32, copy=False),
            "z_fc1": z_fc1.astype(np.float32, copy=False),
            "a_fc1": a_fc1.astype(np.float32, copy=False),
            "logits": logits.astype(np.float32, copy=False),
            "softmax": probs.astype(np.float32, copy=False),
        }

    def debug_compare_forward_pass(self, x_batch: np.ndarray) -> dict[str, Any]:
        """
        Run one identical forward pass on CPU and GPU and compare layer outputs.
        Prints layer stats and returns a structured divergence report.
        """
        n_cur = min(int(self.cfg.debug_batch_size), int(self.cfg.batch_size), int(x_batch.shape[0]))
        if n_cur <= 0:
            raise ValueError("debug_compare_forward_pass requires a non-empty batch.")

        x = np.ascontiguousarray(x_batch[:n_cur].astype(np.float32, copy=False))
        sample_elems = self.cfg.input_channels * self.cfg.input_height * self.cfg.input_width
        buffers = self._alloc_batch_buffers(self.cfg.batch_size)

        # Seed output buffers with NaN to catch partially-written outputs.
        nan_fill = {
            "z_conv": np.full((self.cfg.batch_size, self.cfg.conv_filters, self.oh, self.ow), np.nan, dtype=np.float32),
            "a_conv": np.full((self.cfg.batch_size, self.cfg.conv_filters, self.oh, self.ow), np.nan, dtype=np.float32),
            "z_fc1": np.full((self.cfg.batch_size, self.cfg.hidden_units), np.nan, dtype=np.float32),
            "a_fc1": np.full((self.cfg.batch_size, self.cfg.hidden_units), np.nan, dtype=np.float32),
            "logits": np.full((self.cfg.batch_size, self.cfg.num_classes), np.nan, dtype=np.float32),
        }
        cl.enqueue_copy(self.manager.queue, buffers["z_conv"], nan_fill["z_conv"])
        cl.enqueue_copy(self.manager.queue, buffers["a_conv"], nan_fill["a_conv"])
        cl.enqueue_copy(self.manager.queue, buffers["z_fc1"], nan_fill["z_fc1"])
        cl.enqueue_copy(self.manager.queue, buffers["a_fc1"], nan_fill["a_fc1"])
        cl.enqueue_copy(self.manager.queue, buffers["logits"], nan_fill["logits"])
        self.manager.queue.finish()

        x_dev = self.manager.to_device(x)
        self._forward_only(x_dev=x_dev, n_cur=n_cur, sample_elems=sample_elems, buffers=buffers)

        gpu_intermediates = {
            "z_conv": self.manager.from_device(
                buffers["z_conv"], (n_cur, self.cfg.conv_filters, self.oh, self.ow), np.dtype(np.float32)
            ),
            "a_conv": self.manager.from_device(
                buffers["a_conv"], (n_cur, self.cfg.conv_filters, self.oh, self.ow), np.dtype(np.float32)
            ),
            "z_fc1": self.manager.from_device(buffers["z_fc1"], (n_cur, self.cfg.hidden_units), np.dtype(np.float32)),
            "a_fc1": self.manager.from_device(buffers["a_fc1"], (n_cur, self.cfg.hidden_units), np.dtype(np.float32)),
            "logits": self.manager.from_device(buffers["logits"], (n_cur, self.cfg.num_classes), np.dtype(np.float32)),
        }
        gpu_intermediates["softmax"] = self._stable_softmax(gpu_intermediates["logits"])

        cpu_intermediates = self._cpu_forward_intermediates(x)

        print("\n[DEBUG] CPU vs GPU forward-pass diagnostics")
        print(
            f"[DEBUG] batch={n_cur} rtol={self.cfg.debug_rtol:.2e} atol={self.cfg.debug_atol:.2e} "
            f"input_dtype={x.dtype}"
        )

        params = self._parameter_snapshot()
        gpu_param_dtypes = sorted({str(v.dtype) for v in params.values()})
        cpu_param_dtypes = sorted({str(p.dtype) for p in self.to_cpu_model().parameters()})
        has_float64 = any(dt == "float64" for dt in gpu_param_dtypes + cpu_param_dtypes + [str(x.dtype)])
        print(f"[DEBUG] dtype check: input={x.dtype} cpu_params={cpu_param_dtypes} gpu_params={gpu_param_dtypes}")
        print(f"[DEBUG] float64 mismatch detected: {'YES' if has_float64 else 'NO'}")

        buffer_write_issues: list[str] = []
        for name in ("z_conv", "a_conv", "z_fc1", "a_fc1", "logits"):
            if not np.isfinite(gpu_intermediates[name]).all():
                buffer_write_issues.append(name)
        if buffer_write_issues:
            print(f"[DEBUG] buffer initialization/write issue suspected in: {', '.join(buffer_write_issues)}")
        else:
            print("[DEBUG] buffer initialization/write issue suspected: NO")

        kernel_softmax_stable = "sum_exp += exp(v - max_v)" in KERNEL_SOURCE
        print(f"[DEBUG] softmax stabilization pattern in GPU kernel: {'YES' if kernel_softmax_stable else 'NO'}")

        cpu_overflow = self._exp_overflow_flags(cpu_intermediates["logits"])
        gpu_overflow = self._exp_overflow_flags(gpu_intermediates["logits"])
        print(
            "[DEBUG] exp overflow check (logits): "
            f"CPU would_overflow={cpu_overflow[0]} naive_exp_inf={cpu_overflow[1]} max_logit={cpu_overflow[2]:.6g}; "
            f"GPU would_overflow={gpu_overflow[0]} naive_exp_inf={gpu_overflow[1]} max_logit={gpu_overflow[2]:.6g}"
        )

        layer_order = [
            ("Conv output", "z_conv"),
            ("After ReLU (conv)", "a_conv"),
            ("Dense output (fc1)", "z_fc1"),
            ("After ReLU (fc1)", "a_fc1"),
            ("Dense output (logits)", "logits"),
            ("Softmax output", "softmax"),
        ]

        first_divergence: str | None = None
        layer_report: list[dict[str, Any]] = []
        for title, key in layer_order:
            cpu_arr = cpu_intermediates[key]
            gpu_arr = gpu_intermediates[key]
            diff = np.abs(cpu_arr - gpu_arr)
            allclose = bool(np.allclose(cpu_arr, gpu_arr, rtol=self.cfg.debug_rtol, atol=self.cfg.debug_atol))
            if first_divergence is None and not allclose:
                first_divergence = title

            cpu_stats = self._tensor_stats(cpu_arr)
            gpu_stats = self._tensor_stats(gpu_arr)
            diff_stats = self._tensor_stats(diff)
            print(f"[DEBUG] {title}")
            print(f"        CPU: {self._format_stats(cpu_stats)}")
            print(f"        GPU: {self._format_stats(gpu_stats)}")
            print(f"        |CPU-GPU|: {self._format_stats(diff_stats)} allclose={allclose}")
            layer_report.append(
                {
                    "layer": title,
                    "allclose": allclose,
                    "cpu": cpu_stats,
                    "gpu": gpu_stats,
                    "abs_diff": diff_stats,
                }
            )

        if first_divergence is None:
            print("[DEBUG] First divergence layer: none (within tolerance).")
        else:
            print(f"[DEBUG] First divergence layer: {first_divergence}")

        return {
            "first_divergence_layer": first_divergence,
            "float64_mismatch": has_float64,
            "buffer_write_issues": buffer_write_issues,
            "softmax_stabilization_present": kernel_softmax_stable,
            "exp_overflow": {
                "cpu": {
                    "would_overflow": cpu_overflow[0],
                    "naive_exp_has_inf": cpu_overflow[1],
                    "max_logit": cpu_overflow[2],
                },
                "gpu": {
                    "would_overflow": gpu_overflow[0],
                    "naive_exp_has_inf": gpu_overflow[1],
                    "max_logit": gpu_overflow[2],
                },
            },
            "layers": layer_report,
        }

    def _forward_only(self, x_dev: Any, n_cur: int, sample_elems: int, buffers: dict[str, Any]) -> None:
        cfg = self.cfg
        q = self.manager.queue

        self.k_copy_batch_f32(
            q,
            (int(n_cur * sample_elems),),
            None,
            x_dev,
            buffers["x"],
            np.int32(0),
            np.int32(sample_elems),
            np.int32(n_cur),
        )
        self.k_conv2d_forward_nchw(
            q,
            (int(n_cur), int(cfg.conv_filters), int(self.oh * self.ow)),
            None,
            buffers["x"],
            self.w_conv,
            self.b_conv,
            buffers["z_conv"],
            np.int32(n_cur),
            np.int32(cfg.input_channels),
            np.int32(cfg.input_height),
            np.int32(cfg.input_width),
            np.int32(cfg.conv_filters),
            np.int32(cfg.kernel_size),
            np.int32(cfg.stride),
            np.int32(cfg.padding),
            np.int32(self.oh),
            np.int32(self.ow),
        )
        conv_total = n_cur * cfg.conv_filters * self.oh * self.ow
        self.k_relu_forward(
            q,
            (int(conv_total),),
            None,
            buffers["z_conv"],
            buffers["a_conv"],
            np.int32(conv_total),
        )
        self.k_dense_forward(
            q,
            (int(n_cur), int(cfg.hidden_units)),
            None,
            buffers["a_conv"],
            self.w_fc1,
            self.b_fc1,
            buffers["z_fc1"],
            np.int32(n_cur),
            np.int32(self.flat_features),
            np.int32(cfg.hidden_units),
        )
        fc1_total = n_cur * cfg.hidden_units
        self.k_relu_forward(
            q,
            (int(fc1_total),),
            None,
            buffers["z_fc1"],
            buffers["a_fc1"],
            np.int32(fc1_total),
        )
        self.k_dense_forward(
            q,
            (int(n_cur), int(cfg.num_classes)),
            None,
            buffers["a_fc1"],
            self.w_fc2,
            self.b_fc2,
            buffers["logits"],
            np.int32(n_cur),
            np.int32(cfg.hidden_units),
            np.int32(cfg.num_classes),
        )
        q.finish()

    def sanity_check(self, x_batch: np.ndarray, y_batch: np.ndarray) -> None:
        """
        Fast GPU correctness check:
        - logits should not be all identical
        - one train step should change parameters
        """
        n_cur = min(self.cfg.batch_size, x_batch.shape[0])
        if n_cur <= 0:
            raise ValueError("sanity_check requires non-empty batch.")
        x = np.ascontiguousarray(x_batch[:n_cur].astype(np.float32, copy=False))
        y = np.ascontiguousarray(y_batch[:n_cur].astype(np.int32, copy=False))

        sample_elems = self.cfg.input_channels * self.cfg.input_height * self.cfg.input_width
        buffers = self._alloc_batch_buffers(self.cfg.batch_size)
        x_dev = self.manager.to_device(x)
        y_dev = self.manager.to_device(y)

        # Forward-only validation first to pinpoint stage failures.
        self._forward_only(x_dev=x_dev, n_cur=n_cur, sample_elems=sample_elems, buffers=buffers)
        z_conv = self.manager.from_device(
            buffers["z_conv"], (n_cur, self.cfg.conv_filters, self.oh, self.ow), np.dtype(np.float32)
        )
        a_conv = self.manager.from_device(
            buffers["a_conv"], (n_cur, self.cfg.conv_filters, self.oh, self.ow), np.dtype(np.float32)
        )
        z_fc1 = self.manager.from_device(buffers["z_fc1"], (n_cur, self.cfg.hidden_units), np.dtype(np.float32))
        a_fc1 = self.manager.from_device(buffers["a_fc1"], (n_cur, self.cfg.hidden_units), np.dtype(np.float32))
        logits0 = self.manager.from_device(buffers["logits"], (n_cur, self.cfg.num_classes), np.dtype(np.float32))
        for name, arr in (
            ("z_conv", z_conv),
            ("a_conv", a_conv),
            ("z_fc1", z_fc1),
            ("a_fc1", a_fc1),
            ("logits", logits0),
        ):
            if not np.isfinite(arr).all():
                raise RuntimeError(f"GPU sanity check failed in forward stage. {self._finite_stats(name, arr)}")

        pre = self._parameter_snapshot()
        self._train_batch(
            x_train_dev=x_dev,
            y_train_dev=y_dev,
            start=0,
            n_cur=n_cur,
            sample_elems=sample_elems,
            buffers=buffers,
        )

        logits = self.manager.from_device(buffers["logits"], (n_cur, self.cfg.num_classes), np.dtype(np.float32))
        if not np.isfinite(logits).all():
            w = self._parameter_snapshot()
            stats = (
                f"logits finite={np.isfinite(logits).all()} std={float(np.nanstd(logits)):.6f}; "
                f"w_conv finite={np.isfinite(w['w_conv']).all()} "
                f"w_fc1 finite={np.isfinite(w['w_fc1']).all()} "
                f"w_fc2 finite={np.isfinite(w['w_fc2']).all()}"
            )
            raise RuntimeError(f"GPU sanity check failed: logits contain NaN/Inf. {stats}")
        if float(np.std(logits)) < 1e-7:
            raise RuntimeError("GPU sanity check failed: logits are nearly constant.")

        post = self._parameter_snapshot()
        total_delta = 0.0
        for key in pre:
            total_delta += float(np.linalg.norm(post[key] - pre[key]))
        if total_delta <= 1e-8:
            raise RuntimeError("GPU sanity check failed: parameters did not update.")

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        on_epoch: Callable[[int, float, float, float, dict[str, list[float]]], None] | None = None,
    ) -> dict[str, list[float]]:
        cfg = self.cfg
        if x_train.dtype != np.float32:
            x_train = x_train.astype(np.float32, copy=False)
        if x_test.dtype != np.float32:
            x_test = x_test.astype(np.float32, copy=False)
        if y_train.dtype != np.int32:
            y_train = y_train.astype(np.int32, copy=False)
        if y_test.dtype != np.int32:
            y_test = y_test.astype(np.int32, copy=False)

        if cfg.debug_mode:
            _ = self.debug_compare_forward_pass(x_train)

        x_test_dev = self.manager.to_device(np.ascontiguousarray(x_test))
        y_test_dev = self.manager.to_device(np.ascontiguousarray(y_test))

        batch = cfg.batch_size
        sample_elems = cfg.input_channels * cfg.input_height * cfg.input_width
        buffers = self._alloc_batch_buffers(batch)
        history: dict[str, list[float]] = {"loss": [], "accuracy": [], "epoch_time": []}

        num_train = x_train.shape[0]
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            epoch_seen = 0
            self.k_set_f32_zero(self.manager.queue, (1,), None, buffers["loss_accum"])

            # Shuffle each epoch to avoid ordered-label collapse in SGD.
            if cfg.shuffle:
                perm = np.random.permutation(num_train)
                x_epoch = np.ascontiguousarray(x_train[perm])
                y_epoch = np.ascontiguousarray(y_train[perm])
            else:
                x_epoch = np.ascontiguousarray(x_train)
                y_epoch = np.ascontiguousarray(y_train)
            x_train_dev = self.manager.to_device(x_epoch)
            y_train_dev = self.manager.to_device(y_epoch)

            for start in range(0, num_train, batch):
                n_cur = min(batch, num_train - start)
                self._train_batch(
                    x_train_dev=x_train_dev,
                    y_train_dev=y_train_dev,
                    start=start,
                    n_cur=n_cur,
                    sample_elems=sample_elems,
                    buffers=buffers,
                )

                self.k_reduce_sum_f32_single(
                    self.manager.queue,
                    (1,),
                    None,
                    buffers["losses"],
                    buffers["loss_sum"],
                    np.int32(n_cur),
                )
                self.manager.queue.finish()
                self.k_add_f32_scalar(
                    self.manager.queue,
                    (1,),
                    None,
                    buffers["loss_accum"],
                    buffers["loss_sum"],
                )
                epoch_seen += n_cur

            self.k_div_f32_by_i32(
                self.manager.queue,
                (1,),
                None,
                buffers["loss_accum"],
                buffers["metric_f32"],
                np.int32(max(1, epoch_seen)),
            )
            self.manager.queue.finish()
            avg_loss = float(self.manager.from_device(buffers["metric_f32"], (1,), np.dtype(np.float32))[0])
            accuracy = self.evaluate_accuracy(x_test_dev, y_test_dev, x_test.shape[0], batch, buffers)
            sec = time.perf_counter() - t0

            history["loss"].append(float(avg_loss))
            history["accuracy"].append(float(accuracy))
            history["epoch_time"].append(float(sec))
            if on_epoch is not None:
                on_epoch(epoch, float(avg_loss), float(accuracy), float(sec), history)

        return history

    def _train_batch(
        self,
        x_train_dev: Any,
        y_train_dev: Any,
        start: int,
        n_cur: int,
        sample_elems: int,
        buffers: dict[str, Any],
    ) -> None:
        cfg = self.cfg
        q = self.manager.queue

        self.k_copy_batch_f32(
            q,
            (int(n_cur * sample_elems),),
            None,
            x_train_dev,
            buffers["x"],
            np.int32(start),
            np.int32(sample_elems),
            np.int32(n_cur),
        )
        self.k_copy_batch_i32(
            q,
            (int(n_cur),),
            None,
            y_train_dev,
            buffers["y"],
            np.int32(start),
            np.int32(n_cur),
        )

        self.k_conv2d_forward_nchw(
            q,
            (int(n_cur), int(cfg.conv_filters), int(self.oh * self.ow)),
            None,
            buffers["x"],
            self.w_conv,
            self.b_conv,
            buffers["z_conv"],
            np.int32(n_cur),
            np.int32(cfg.input_channels),
            np.int32(cfg.input_height),
            np.int32(cfg.input_width),
            np.int32(cfg.conv_filters),
            np.int32(cfg.kernel_size),
            np.int32(cfg.stride),
            np.int32(cfg.padding),
            np.int32(self.oh),
            np.int32(self.ow),
        )

        conv_total = n_cur * cfg.conv_filters * self.oh * self.ow
        self.k_relu_forward(
            q,
            (int(conv_total),),
            None,
            buffers["z_conv"],
            buffers["a_conv"],
            np.int32(conv_total),
        )

        self.k_dense_forward(
            q,
            (int(n_cur), int(cfg.hidden_units)),
            None,
            buffers["a_conv"],
            self.w_fc1,
            self.b_fc1,
            buffers["z_fc1"],
            np.int32(n_cur),
            np.int32(self.flat_features),
            np.int32(cfg.hidden_units),
        )

        fc1_total = n_cur * cfg.hidden_units
        self.k_relu_forward(
            q,
            (int(fc1_total),),
            None,
            buffers["z_fc1"],
            buffers["a_fc1"],
            np.int32(fc1_total),
        )

        self.k_dense_forward(
            q,
            (int(n_cur), int(cfg.num_classes)),
            None,
            buffers["a_fc1"],
            self.w_fc2,
            self.b_fc2,
            buffers["logits"],
            np.int32(n_cur),
            np.int32(cfg.hidden_units),
            np.int32(cfg.num_classes),
        )

        self.k_softmax_xent_grad(
            q,
            (int(n_cur),),
            None,
            buffers["logits"],
            buffers["y"],
            buffers["d_logits"],
            buffers["losses"],
            np.int32(n_cur),
            np.int32(cfg.num_classes),
        )

        self.k_dense_backward_weight(
            q,
            (int(cfg.hidden_units), int(cfg.num_classes)),
            None,
            buffers["a_fc1"],
            buffers["d_logits"],
            self.gw_fc2,
            np.int32(n_cur),
            np.int32(cfg.hidden_units),
            np.int32(cfg.num_classes),
        )
        self.k_dense_backward_bias(
            q,
            (int(cfg.num_classes),),
            None,
            buffers["d_logits"],
            self.gb_fc2,
            np.int32(n_cur),
            np.int32(cfg.num_classes),
        )
        self.k_dense_backward_input(
            q,
            (int(n_cur), int(cfg.hidden_units)),
            None,
            buffers["d_logits"],
            self.w_fc2,
            buffers["d_a_fc1"],
            np.int32(n_cur),
            np.int32(cfg.hidden_units),
            np.int32(cfg.num_classes),
        )

        self.k_relu_backward(
            q,
            (int(fc1_total),),
            None,
            buffers["d_a_fc1"],
            buffers["z_fc1"],
            buffers["d_z_fc1"],
            np.int32(fc1_total),
        )

        self.k_dense_backward_weight(
            q,
            (int(self.flat_features), int(cfg.hidden_units)),
            None,
            buffers["a_conv"],
            buffers["d_z_fc1"],
            self.gw_fc1,
            np.int32(n_cur),
            np.int32(self.flat_features),
            np.int32(cfg.hidden_units),
        )
        self.k_dense_backward_bias(
            q,
            (int(cfg.hidden_units),),
            None,
            buffers["d_z_fc1"],
            self.gb_fc1,
            np.int32(n_cur),
            np.int32(cfg.hidden_units),
        )
        self.k_dense_backward_input(
            q,
            (int(n_cur), int(self.flat_features)),
            None,
            buffers["d_z_fc1"],
            self.w_fc1,
            buffers["d_a_conv"],
            np.int32(n_cur),
            np.int32(self.flat_features),
            np.int32(cfg.hidden_units),
        )

        self.k_relu_backward(
            q,
            (int(conv_total),),
            None,
            buffers["d_a_conv"],
            buffers["z_conv"],
            buffers["d_z_conv"],
            np.int32(conv_total),
        )

        self.k_conv2d_backward_weight_nchw(
            q,
            (int(cfg.conv_filters), int(cfg.input_channels), int(cfg.kernel_size * cfg.kernel_size)),
            None,
            buffers["x"],
            buffers["d_z_conv"],
            self.gw_conv,
            np.int32(n_cur),
            np.int32(cfg.input_channels),
            np.int32(cfg.input_height),
            np.int32(cfg.input_width),
            np.int32(cfg.conv_filters),
            np.int32(cfg.kernel_size),
            np.int32(cfg.stride),
            np.int32(cfg.padding),
            np.int32(self.oh),
            np.int32(self.ow),
        )
        self.k_conv2d_backward_bias_nchw(
            q,
            (int(cfg.conv_filters),),
            None,
            buffers["d_z_conv"],
            self.gb_conv,
            np.int32(n_cur),
            np.int32(cfg.conv_filters),
            np.int32(self.oh),
            np.int32(self.ow),
        )
        self.k_conv2d_backward_input_nchw(
            q,
            (int(n_cur), int(cfg.input_channels), int(cfg.input_height * cfg.input_width)),
            None,
            buffers["d_z_conv"],
            self.w_conv,
            buffers["d_x"],
            np.int32(n_cur),
            np.int32(cfg.input_channels),
            np.int32(cfg.input_height),
            np.int32(cfg.input_width),
            np.int32(cfg.conv_filters),
            np.int32(cfg.kernel_size),
            np.int32(cfg.stride),
            np.int32(cfg.padding),
            np.int32(self.oh),
            np.int32(self.ow),
        )

        self._sgd_step(cfg.learning_rate)
        q.finish()

    def _sgd_step(self, lr: float) -> None:
        q = self.manager.queue
        params_and_grads = [
            (self.w_conv, self.gw_conv, self.cfg.conv_filters * self.cfg.input_channels * self.cfg.kernel_size * self.cfg.kernel_size),
            (self.b_conv, self.gb_conv, self.cfg.conv_filters),
            (self.w_fc1, self.gw_fc1, self.flat_features * self.cfg.hidden_units),
            (self.b_fc1, self.gb_fc1, self.cfg.hidden_units),
            (self.w_fc2, self.gw_fc2, self.cfg.hidden_units * self.cfg.num_classes),
            (self.b_fc2, self.gb_fc2, self.cfg.num_classes),
        ]
        for param, grad, n in params_and_grads:
            self.k_sgd_update(
                q,
                (int(n),),
                None,
                param,
                grad,
                np.float32(lr),
                np.int32(n),
            )

    def evaluate_accuracy(
        self,
        x_dev: Any,
        y_dev: Any,
        total_samples: int,
        batch_size: int,
        buffers: dict[str, Any],
    ) -> float:
        cfg = self.cfg
        sample_elems = cfg.input_channels * cfg.input_height * cfg.input_width
        self.k_set_f32_zero(self.manager.queue, (1,), None, buffers["correct_accum"])

        for start in range(0, total_samples, batch_size):
            n_cur = min(batch_size, total_samples - start)
            self.k_copy_batch_f32(
                self.manager.queue,
                (int(n_cur * sample_elems),),
                None,
                x_dev,
                buffers["x"],
                np.int32(start),
                np.int32(sample_elems),
                np.int32(n_cur),
            )
            self.k_copy_batch_i32(
                self.manager.queue,
                (int(n_cur),),
                None,
                y_dev,
                buffers["y"],
                np.int32(start),
                np.int32(n_cur),
            )

            self.k_conv2d_forward_nchw(
                self.manager.queue,
                (int(n_cur), int(cfg.conv_filters), int(self.oh * self.ow)),
                None,
                buffers["x"],
                self.w_conv,
                self.b_conv,
                buffers["z_conv"],
                np.int32(n_cur),
                np.int32(cfg.input_channels),
                np.int32(cfg.input_height),
                np.int32(cfg.input_width),
                np.int32(cfg.conv_filters),
                np.int32(cfg.kernel_size),
                np.int32(cfg.stride),
                np.int32(cfg.padding),
                np.int32(self.oh),
                np.int32(self.ow),
            )
            conv_total = n_cur * cfg.conv_filters * self.oh * self.ow
            self.k_relu_forward(
                self.manager.queue,
                (int(conv_total),),
                None,
                buffers["z_conv"],
                buffers["a_conv"],
                np.int32(conv_total),
            )

            self.k_dense_forward(
                self.manager.queue,
                (int(n_cur), int(cfg.hidden_units)),
                None,
                buffers["a_conv"],
                self.w_fc1,
                self.b_fc1,
                buffers["z_fc1"],
                np.int32(n_cur),
                np.int32(self.flat_features),
                np.int32(cfg.hidden_units),
            )
            fc1_total = n_cur * cfg.hidden_units
            self.k_relu_forward(
                self.manager.queue,
                (int(fc1_total),),
                None,
                buffers["z_fc1"],
                buffers["a_fc1"],
                np.int32(fc1_total),
            )
            self.k_dense_forward(
                self.manager.queue,
                (int(n_cur), int(cfg.num_classes)),
                None,
                buffers["a_fc1"],
                self.w_fc2,
                self.b_fc2,
                buffers["logits"],
                np.int32(n_cur),
                np.int32(cfg.hidden_units),
                np.int32(cfg.num_classes),
            )

            self.k_accuracy_flag_from_logits(
                self.manager.queue,
                (int(n_cur),),
                None,
                buffers["logits"],
                buffers["y"],
                buffers["correct_flags"],
                np.int32(n_cur),
                np.int32(cfg.num_classes),
            )
            self.k_reduce_sum_f32_single(
                self.manager.queue,
                (1,),
                None,
                buffers["correct_flags"],
                buffers["correct_sum"],
                np.int32(n_cur),
            )
            self.k_add_f32_scalar(
                self.manager.queue,
                (1,),
                None,
                buffers["correct_accum"],
                buffers["correct_sum"],
            )

        self.k_div_f32_by_i32(
            self.manager.queue,
            (1,),
            None,
            buffers["correct_accum"],
            buffers["metric_f32"],
            np.int32(max(1, total_samples)),
        )
        self.manager.queue.finish()
        return float(self.manager.from_device(buffers["metric_f32"], (1,), np.dtype(np.float32))[0])

    def to_cpu_model(self) -> CNNModel:
        """Materialize current GPU weights into a CPU CNNModel instance."""
        cfg = self.cfg
        model = CNNModel(
            [
                Conv2D(in_channels=1, out_channels=cfg.conv_filters, kernel_size=3, stride=1, padding=0),
                ReLU(),
                Flatten(),
                Dense(in_features=self.flat_features, out_features=cfg.hidden_units),
                ReLU(),
                Dense(in_features=cfg.hidden_units, out_features=cfg.num_classes),
            ]
        )
        params = model.parameters()
        if len(params) != 6:
            raise RuntimeError(f"Unexpected parameter count in CPU model: {len(params)}.")

        host_w_conv = self.manager.from_device(
            self.w_conv,
            (cfg.conv_filters, cfg.input_channels, cfg.kernel_size, cfg.kernel_size),
            np.dtype(np.float32),
        )
        host_b_conv = self.manager.from_device(self.b_conv, (cfg.conv_filters,), np.dtype(np.float32))
        host_w_fc1 = self.manager.from_device(
            self.w_fc1,
            (self.flat_features, cfg.hidden_units),
            np.dtype(np.float32),
        )
        host_b_fc1 = self.manager.from_device(self.b_fc1, (cfg.hidden_units,), np.dtype(np.float32))
        host_w_fc2 = self.manager.from_device(
            self.w_fc2,
            (cfg.hidden_units, cfg.num_classes),
            np.dtype(np.float32),
        )
        host_b_fc2 = self.manager.from_device(self.b_fc2, (cfg.num_classes,), np.dtype(np.float32))

        params[0][...] = host_w_conv
        params[1][...] = host_b_conv
        params[2][...] = host_w_fc1
        params[3][...] = host_b_fc1
        params[4][...] = host_w_fc2
        params[5][...] = host_b_fc2
        return model

    def save_weights_npz(self, file_path: str | Path) -> Path:
        """Save current GPU model weights in the same format as CNNModel.save_weights."""
        path = Path(file_path)
        cpu_model = self.to_cpu_model()
        cpu_model.save_weights(path)
        return path

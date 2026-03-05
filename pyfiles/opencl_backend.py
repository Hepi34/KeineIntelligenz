# Hepi34, onatic07, fritziii, 2026-03-06

"""PyOpenCL setup helpers for device detection and tensor buffer transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pyopencl as cl
except Exception:  # noqa: BLE001
    cl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GPUDeviceInfo:
    platform_name: str
    device_name: str
    vendor: str
    driver_version: str


class OpenCLManager:
    """Owns OpenCL context/queue and utility methods for tensor transfers."""

    def __init__(self, context: Any, queue: Any, device: Any, info: GPUDeviceInfo) -> None:
        self.context = context
        self.queue = queue
        self.device = device
        self.info = info

    @classmethod
    def create(cls) -> OpenCLManager | None:
        """Detect a GPU device and create context + command queue."""
        if cl is None:
            return None

        try:
            for platform in cl.get_platforms():
                gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
                if not gpu_devices:
                    continue

                device = gpu_devices[0]
                context = cl.Context(devices=[device])
                queue = cl.CommandQueue(context, device=device)
                info = GPUDeviceInfo(
                    platform_name=str(platform.name),
                    device_name=str(device.name),
                    vendor=str(device.vendor),
                    driver_version=str(device.driver_version),
                )
                return cls(context=context, queue=queue, device=device, info=info)
        except Exception:  # noqa: BLE001
            return None

        return None

    def to_device(self, array: np.ndarray) -> Any:
        """Copy a NumPy tensor to a GPU buffer."""
        if cl is None:
            raise RuntimeError("PyOpenCL is not available.")
        host = np.ascontiguousarray(array)
        # Allocate buffer WITHOUT relying on COPY_HOST_PTR, which may have issues
        flags = cl.mem_flags.READ_WRITE
        buffer = cl.Buffer(self.context, flags, size=host.nbytes)
        # Explicitly copy data using enqueue_copy
        cl.enqueue_copy(self.queue, buffer, host)
        self.queue.finish()
        return buffer

    def empty_device(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> Any:
        """Allocate an empty GPU buffer matching shape/dtype."""
        if cl is None:
            raise RuntimeError("PyOpenCL is not available.")
        size_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        flags = cl.mem_flags.READ_WRITE
        return cl.Buffer(self.context, flags, size=size_bytes)

    def from_device(self, buffer: Any, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
        """Copy GPU buffer contents back to CPU NumPy array."""
        if cl is None:
            raise RuntimeError("PyOpenCL is not available.")
        out = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.queue, out, buffer)
        self.queue.finish()
        return out

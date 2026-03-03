"""CNN model definition and orchestration logic."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from layers import Layer


class CNNModel:
    """Composable CNN model made of layer objects."""

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[np.ndarray]:
        params: list[np.ndarray] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def gradients(self) -> list[np.ndarray]:
        grads: list[np.ndarray] = []
        for layer in self.layers:
            grads.extend(layer.gradients())
        return grads

    def iter_parameter_pairs(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        """Yield (param, grad) pairs in optimizer-ready format."""
        for layer in self.layers:
            for param, grad in zip(layer.parameters(), layer.gradients(), strict=False):
                yield param, grad

    def save_weights(self, file_path: str | Path, metadata: dict[str, str] | None = None) -> None:
        """Save all trainable parameters to a NumPy .npz file."""
        params = self.parameters()
        payload = {f"param_{idx}": param for idx, param in enumerate(params)}
        payload["num_params"] = np.array([len(params)], dtype=np.int64)
        if metadata is not None:
            for key, value in metadata.items():
                payload[f"meta_{key}"] = np.array([str(value)])
        np.savez(Path(file_path), **payload)

    @staticmethod
    def load_metadata(file_path: str | Path) -> dict[str, str]:
        """Read optional metadata fields from a NumPy .npz weights file."""
        path = Path(file_path)
        out: dict[str, str] = {}
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if key.startswith("meta_"):
                    value = data[key]
                    if value.size > 0:
                        out[key[5:]] = str(value.reshape(-1)[0])
        return out

    def load_weights(self, file_path: str | Path) -> None:
        """Load trainable parameters from a NumPy .npz file into this model."""
        path = Path(file_path)
        with np.load(path, allow_pickle=False) as data:
            params = self.parameters()
            if "num_params" in data:
                expected = int(data["num_params"][0])
                if expected != len(params):
                    raise ValueError(
                        f"Parameter count mismatch: file has {expected}, model expects {len(params)}."
                    )

            for idx, param in enumerate(params):
                key = f"param_{idx}"
                if key not in data:
                    raise ValueError(f"Missing parameter '{key}' in {path}.")
                loaded = data[key]
                if loaded.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: file {loaded.shape}, model {param.shape}."
                    )
                param[...] = loaded

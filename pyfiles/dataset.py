"""MNIST dataset loading and normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MNISTDataset:
    """Container for MNIST train/test splits."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_idx_images(file_path: str | Path) -> np.ndarray:
    """Load an IDX image file into shape (N, 1, 28, 28), dtype float32 in [0, 255]."""
    path = Path(file_path)
    raw = path.read_bytes()

    # IDX image format (big-endian):
    # bytes 0-3   : magic number (2051 for images)
    # bytes 4-7   : number of images
    # bytes 8-11  : number of rows
    # bytes 12-15 : number of columns
    # bytes 16..  : pixel data (unsigned byte per pixel)
    magic = int.from_bytes(raw[0:4], byteorder="big")
    if magic != 2051:
        raise ValueError(f"Invalid image magic number in {path}: {magic} (expected 2051).")

    num_images = int.from_bytes(raw[4:8], byteorder="big")
    rows = int.from_bytes(raw[8:12], byteorder="big")
    cols = int.from_bytes(raw[12:16], byteorder="big")

    expected_pixels = num_images * rows * cols
    pixel_data = np.frombuffer(raw, dtype=np.uint8, offset=16)
    if pixel_data.size != expected_pixels:
        raise ValueError(
            f"Image file size mismatch in {path}: found {pixel_data.size} pixels, "
            f"expected {expected_pixels}."
        )

    images = pixel_data.reshape(num_images, 1, rows, cols).astype(np.float32)
    return images


def load_idx_labels(file_path: str | Path) -> np.ndarray:
    """Load an IDX label file into shape (N,), dtype int64."""
    path = Path(file_path)
    raw = path.read_bytes()

    # IDX label format (big-endian):
    # bytes 0-3  : magic number (2049 for labels)
    # bytes 4-7  : number of labels
    # bytes 8..  : label data (unsigned byte per label, values 0-9 for MNIST)
    magic = int.from_bytes(raw[0:4], byteorder="big")
    if magic != 2049:
        raise ValueError(f"Invalid label magic number in {path}: {magic} (expected 2049).")

    num_labels = int.from_bytes(raw[4:8], byteorder="big")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    if labels.size != num_labels:
        raise ValueError(
            f"Label file size mismatch in {path}: found {labels.size} labels, "
            f"expected {num_labels}."
        )

    return labels.astype(np.int64)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize raw image values to [0, 1]."""
    return images.astype(np.float32) / 255.0


def load_mnist(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and normalize MNIST from raw ubyte files.

    Returns:
        X_train, y_train, X_test, y_test
    """
    data_path = Path(data_dir)

    train_images_path = data_path / "train-images.idx3.ubyte"
    train_labels_path = data_path / "train-labels.idx1.ubyte"
    test_images_path = data_path / "t10k-images.idx3.ubyte"
    test_labels_path = data_path / "t10k-labels.idx1.ubyte"
    return load_mnist_from_files(
        train_images_path=train_images_path,
        train_labels_path=train_labels_path,
        test_images_path=test_images_path,
        test_labels_path=test_labels_path,
    )


def load_mnist_from_files(
    train_images_path: str | Path,
    train_labels_path: str | Path,
    test_images_path: str | Path,
    test_labels_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and normalize MNIST from explicit file paths."""
    x_train = normalize_images(load_idx_images(train_images_path))
    y_train = load_idx_labels(train_labels_path)
    x_test = normalize_images(load_idx_images(test_images_path))
    y_test = load_idx_labels(test_labels_path)

    if x_train.shape[1:] != (1, 28, 28) or x_test.shape[1:] != (1, 28, 28):
        raise ValueError(
            "Unexpected image shape. Expected MNIST image shape (N, 1, 28, 28), "
            f"got train={x_train.shape}, test={x_test.shape}."
        )

    return x_train, y_train, x_test, y_test

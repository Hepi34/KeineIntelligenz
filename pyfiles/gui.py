# Hepi34, onatic07, fritziii, 2026-03-06

"""PyQt6 GUI for MNIST CNN training."""

from __future__ import annotations

import sys
import time
import os
import re
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QPoint, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QImage, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dataset import load_mnist_from_files
from layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, ReLU
from loss import CrossEntropy
from model import CNNModel
from opencl_backend import OpenCLManager
from optimizers import Adam, SGD
from gpu_pipeline import GPUTrainConfig, GPUTrainingPipeline
from trainer import TrainConfig, Trainer


@dataclass(frozen=True)
class Preset:
    key: str
    version: str
    name: str
    epochs: int
    batch_size: int
    lr: float
    conv_filters: int
    hidden_units: int
    train_limit: int
    test_limit: int
    conv2_filters: int | None = None
    optimizer: str = "sgd"
    use_second_conv: bool = False
    use_maxpool: bool = False
    dropout_rate: float = 0.0
    weight_decay: float = 0.0
    lr_decay_after_epoch: int | None = None
    lr_decay_factor: float = 1.0
    restore_best: bool = False


@dataclass(frozen=True)
class MNISTFilePaths:
    train_images: Path
    train_labels: Path
    test_images: Path
    test_labels: Path


PRESETS: dict[str, Preset] = {
    # v1 (original)
    "v1/Mini": Preset(
        key="v1/Mini",
        version="v1",
        name="Mini",
        epochs=2,
        batch_size=64,
        lr=0.03,
        conv_filters=8,
        hidden_units=64,
        train_limit=2000,
        test_limit=1000,
    ),
    "v1/Normal": Preset(
        key="v1/Normal",
        version="v1",
        name="Normal",
        epochs=3,
        batch_size=64,
        lr=0.02,
        conv_filters=12,
        hidden_units=96,
        train_limit=10000,
        test_limit=2000,
    ),
    "v1/Pro": Preset(
        key="v1/Pro",
        version="v1",
        name="Pro",
        epochs=5,
        batch_size=64,
        lr=0.01,
        conv_filters=16,
        hidden_units=128,
        train_limit=60000,
        test_limit=10000,
    ),
    # v2 (tuned for better accuracy)
    "v2/Mini": Preset(
        key="v2/Mini",
        version="v2",
        name="Mini",
        epochs=4,
        batch_size=64,
        lr=0.015,
        conv_filters=12,
        hidden_units=96,
        train_limit=4000,
        test_limit=1000,
    ),
    "v2/Normal": Preset(
        key="v2/Normal",
        version="v2",
        name="Normal",
        epochs=6,
        batch_size=64,
        lr=0.008,
        conv_filters=20,
        hidden_units=160,
        train_limit=20000,
        test_limit=5000,
    ),
    "v2/Pro": Preset(
        key="v2/Pro",
        version="v2",
        name="Pro",
        epochs=10,
        batch_size=64,
        lr=0.005,
        conv_filters=24,
        hidden_units=192,
        train_limit=60000,
        test_limit=10000,
    ),
    # v3 (higher-capacity presets)
    "v3/Mini": Preset(
        key="v3/Mini",
        version="v3",
        name="Mini",
        epochs=3,
        batch_size=64,
        lr=0.01,
        conv_filters=16,
        hidden_units=96,
        train_limit=6000,
        test_limit=2000,
    ),
    "v3/Normal": Preset(
        key="v3/Normal",
        version="v3",
        name="Normal",
        epochs=6,
        batch_size=64,
        lr=0.007,
        conv_filters=24,
        hidden_units=160,
        train_limit=25000,
        test_limit=5000,
    ),
    "v3/Pro": Preset(
        key="v3/Pro",
        version="v3",
        name="Pro",
        epochs=10,
        batch_size=64,
        lr=0.0045,
        conv_filters=32,
        hidden_units=256,
        train_limit=60000,
        test_limit=10000,
    ),
    "v3/Extreme": Preset(
        key="v3/Extreme",
        version="v3",
        name="Extreme",
        epochs=14,
        batch_size=64,
        lr=0.0035,
        conv_filters=48,
        hidden_units=384,
        train_limit=60000,
        test_limit=10000,
    ),
    # v4 (2xConv + MaxPool + Adam)
    "v4/Mini": Preset(
        key="v4/Mini",
        version="v4",
        name="Mini",
        epochs=4,
        batch_size=64,
        lr=0.0015,
        conv_filters=16,
        conv2_filters=24,
        hidden_units=128,
        train_limit=12000,
        test_limit=2000,
        optimizer="adam",
        use_second_conv=True,
        use_maxpool=True,
    ),
    "v4/Normal": Preset(
        key="v4/Normal",
        version="v4",
        name="Normal",
        epochs=8,
        batch_size=64,
        lr=0.0010,
        conv_filters=24,
        conv2_filters=36,
        hidden_units=192,
        train_limit=30000,
        test_limit=5000,
        optimizer="adam",
        use_second_conv=True,
        use_maxpool=True,
    ),
    "v4/Pro": Preset(
        key="v4/Pro",
        version="v4",
        name="Pro",
        epochs=12,
        batch_size=64,
        lr=0.0008,
        conv_filters=32,
        conv2_filters=48,
        hidden_units=256,
        train_limit=60000,
        test_limit=10000,
        optimizer="adam",
        use_second_conv=True,
        use_maxpool=True,
    ),
    "v4/Extreme": Preset(
        key="v4/Extreme",
        version="v4",
        name="Extreme",
        epochs=16,
        batch_size=64,
        lr=0.0006,
        conv_filters=40,
        conv2_filters=64,
        hidden_units=320,
        train_limit=60000,
        test_limit=10000,
        optimizer="adam",
        use_second_conv=True,
        use_maxpool=True,
    ),
    # v5 (2xConv + MaxPool + Adam) tuned for fast and efficient 99%+
    "v5/Mini": Preset(
        key="v5/Mini",
        version="v5",
        name="Mini",
        epochs=3,
        batch_size=64,
        lr=0.0010,
        conv_filters=16,
        conv2_filters=24,
        hidden_units=128,
        train_limit=12000,
        test_limit=2000,
        optimizer="adam",
        use_second_conv=True,
        use_maxpool=True,
        dropout_rate=0.30,
        weight_decay=1e-4,
        lr_decay_after_epoch=2,
        lr_decay_factor=0.5,
        restore_best=True,
    ),
    "v5/Normal": Preset(
        key="v5/Normal",
        version="v5",
        name="Normal",
        epochs=3,
        batch_size=96,
        lr=0.0010,
        conv_filters=24,
        conv2_filters=36,
        hidden_units=176,
        train_limit=28000,
        test_limit=10000,
        optimizer="adam",
        use_second_conv=True,
        use_maxpool=True,
        dropout_rate=0.28,
        weight_decay=1e-4,
        lr_decay_after_epoch=2,
        lr_decay_factor=0.5,
        restore_best=True,
    ),
    "v5/Pro": Preset(
        key="v5/Pro",
        version="v5",
        name="Pro",
        epochs=4,
        batch_size=96,
        lr=0.0009,
        conv_filters=28,
        conv2_filters=44,
        hidden_units=224,
        train_limit=50000,
        test_limit=10000,
        optimizer="adam",
        use_second_conv=True,
        use_maxpool=True,
        dropout_rate=0.24,
        weight_decay=8e-5,
        lr_decay_after_epoch=2,
        lr_decay_factor=0.5,
        restore_best=True,
    ),
    "v5/Extreme": Preset(
        key="v5/Extreme",
        version="v5",
        name="Extreme",
        epochs=5,
        batch_size=128,
        lr=0.0008,
        conv_filters=32,
        conv2_filters=48,
        hidden_units=256,
        train_limit=60000,
        test_limit=10000,
        optimizer="adam",
        use_second_conv=True,
        use_maxpool=True,
        dropout_rate=0.18,
        weight_decay=5e-5,
        lr_decay_after_epoch=2,
        lr_decay_factor=0.5,
        restore_best=True,
    ),
}


def _sanitize_hardware_name(name: str) -> str:
    """Convert hardware names like 'RTX 5080' into filename-safe 'RTX5080'."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", name)
    return cleaned or "Unknown"


def _detect_cpu_name() -> str:
    """Best-effort CPU marketing name for filename labeling."""
    candidates: list[str] = []

    # macOS provides the most useful brand string through sysctl.
    if sys.platform == "darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if brand:
                candidates.append(brand)
        except Exception:
            pass

        # Apple Silicon fallback where brand string can be generic/empty.
        try:
            arm_brand = subprocess.check_output(
                ["sysctl", "-n", "hw.optional.arm64"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if arm_brand == "1":
                candidates.append("Apple Silicon")
        except Exception:
            pass

    candidates.extend(
        [
            platform.processor(),
            platform.machine(),
        ]
    )

    for candidate in candidates:
        if candidate and candidate.lower() not in {"unknown", "arm64", "x86_64"}:
            return candidate
    for candidate in candidates:
        if candidate:
            return candidate
    return "UnknownCPU"


def _format_eta_seconds(total_seconds: float) -> str:
    """Format remaining time as a compact ETA string."""
    secs = max(0, int(round(total_seconds)))
    hours, rem = divmod(secs, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def build_model(preset: Preset) -> CNNModel:
    # v4: Conv3x3 -> ReLU -> Conv3x3 -> ReLU -> MaxPool2x2 -> Flatten -> Dense -> ReLU -> Dense
    if preset.use_second_conv:
        conv2_filters = int(preset.conv2_filters or (preset.conv_filters * 2))
        c1_h = 26  # 28 -> conv3x3 valid
        c2_h = c1_h - 2  # second conv3x3 valid
        post_h = c2_h // 2 if preset.use_maxpool else c2_h
        flattened = conv2_filters * post_h * post_h
        layers: list = [
            Conv2D(in_channels=1, out_channels=preset.conv_filters, kernel_size=3, stride=1, padding=0),
            ReLU(),
            Conv2D(in_channels=preset.conv_filters, out_channels=conv2_filters, kernel_size=3, stride=1, padding=0),
            ReLU(),
        ]
        if preset.use_maxpool:
            layers.append(MaxPool2D(kernel_size=2, stride=2))
        layers.extend(
            [
                Flatten(),
                Dense(in_features=flattened, out_features=preset.hidden_units),
                ReLU(),
            ]
        )
        if preset.dropout_rate > 0.0:
            layers.append(Dropout(rate=preset.dropout_rate))
        layers.append(Dense(in_features=preset.hidden_units, out_features=10))
        return CNNModel(layers)

    # v1-v3 baseline path.
    flattened = preset.conv_filters * 26 * 26
    return CNNModel(
        [
            Conv2D(in_channels=1, out_channels=preset.conv_filters, kernel_size=3, stride=1, padding=0),
            ReLU(),
            Flatten(),
            Dense(in_features=flattened, out_features=preset.hidden_units),
            ReLU(),
            Dense(in_features=preset.hidden_units, out_features=10),
        ]
    )


class TrainingWorker(QObject):
    """Background worker to keep the GUI responsive."""

    progress = pyqtSignal(int, float, float, float, object)  # epoch, loss, accuracy, sec, history
    finished = pyqtSignal(object, object)  # history, model
    failed = pyqtSignal(str)

    def __init__(self, preset: Preset, num_threads: int, paths: MNISTFilePaths) -> None:
        super().__init__()
        self.preset = preset
        self.num_threads = num_threads
        self.paths = paths

    def run(self) -> None:
        try:
            x_train, y_train, x_test, y_test = load_mnist_from_files(
                train_images_path=self.paths.train_images,
                train_labels_path=self.paths.train_labels,
                test_images_path=self.paths.test_images,
                test_labels_path=self.paths.test_labels,
            )
            x_train = x_train[: self.preset.train_limit]
            y_train = y_train[: self.preset.train_limit]
            x_test = x_test[: self.preset.test_limit]
            y_test = y_test[: self.preset.test_limit]

            model = build_model(self.preset)
            loss_fn = CrossEntropy()
            optimizer = (
                Adam(lr=self.preset.lr, weight_decay=self.preset.weight_decay)
                if self.preset.optimizer.lower() == "adam"
                else SGD(lr=self.preset.lr, weight_decay=self.preset.weight_decay)
            )
            trainer = Trainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                config=TrainConfig(
                    epochs=self.preset.epochs,
                    batch_size=self.preset.batch_size,
                    shuffle=True,
                    num_threads=self.num_threads,
                ),
            )

            history: dict[str, list[float]] = {"loss": [], "accuracy": [], "epoch_time": []}
            base_lr = float(self.preset.lr)
            best_acc = -1.0
            best_params: list[np.ndarray] | None = None

            for epoch in range(1, self.preset.epochs + 1):
                if hasattr(optimizer, "lr"):
                    if self.preset.lr_decay_after_epoch is not None and epoch > self.preset.lr_decay_after_epoch:
                        decay_steps = epoch - self.preset.lr_decay_after_epoch
                        optimizer.lr = float(base_lr * (self.preset.lr_decay_factor ** decay_steps))
                    else:
                        optimizer.lr = float(base_lr)
                start = time.perf_counter()
                train_metrics = trainer.train_epoch(x_train, y_train)
                eval_metrics = trainer.evaluate(x_test, y_test)
                sec = time.perf_counter() - start

                loss = float(train_metrics["loss"])
                acc = float(eval_metrics["accuracy"])
                history["loss"].append(loss)
                history["accuracy"].append(acc)
                history["epoch_time"].append(sec)
                if self.preset.restore_best and acc > best_acc:
                    best_acc = acc
                    best_params = [param.copy() for param in model.parameters()]

                self.progress.emit(epoch, loss, acc, sec, history.copy())

            if self.preset.restore_best and best_params is not None:
                for param, best in zip(model.parameters(), best_params, strict=True):
                    param[...] = best
            self.finished.emit(history, model)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class GPUTrainingWorker(QObject):
    """Background worker for fully GPU-based OpenCL training."""

    progress = pyqtSignal(int, float, float, float, object)  # epoch, loss, accuracy, sec, history
    finished = pyqtSignal(object, object)  # history, model(None for GPU path)
    failed = pyqtSignal(str)

    def __init__(self, preset: Preset, paths: MNISTFilePaths, opencl_manager: OpenCLManager) -> None:
        super().__init__()
        self.preset = preset
        self.paths = paths
        self.opencl_manager = opencl_manager

    def run(self) -> None:
        try:
            x_train, y_train, x_test, y_test = load_mnist_from_files(
                train_images_path=self.paths.train_images,
                train_labels_path=self.paths.train_labels,
                test_images_path=self.paths.test_images,
                test_labels_path=self.paths.test_labels,
            )
            x_train = x_train[: self.preset.train_limit].astype(np.float32, copy=False)
            y_train = y_train[: self.preset.train_limit].astype(np.int32, copy=False)
            x_test = x_test[: self.preset.test_limit].astype(np.float32, copy=False)
            y_test = y_test[: self.preset.test_limit].astype(np.int32, copy=False)

            # One-time shuffle before upload to avoid label-order effects on SGD.
            perm = np.random.permutation(x_train.shape[0])
            x_train = x_train[perm]
            y_train = y_train[perm]

            pipeline = GPUTrainingPipeline(
                manager=self.opencl_manager,
                config=GPUTrainConfig(
                    epochs=self.preset.epochs,
                    batch_size=self.preset.batch_size,
                    learning_rate=self.preset.lr,
                    conv_filters=self.preset.conv_filters,
                    conv2_filters=int(self.preset.conv2_filters or 0),
                    hidden_units=self.preset.hidden_units,
                    use_second_conv=self.preset.use_second_conv,
                    use_maxpool=self.preset.use_maxpool,
                    dropout_rate=self.preset.dropout_rate,
                    optimizer=self.preset.optimizer,
                    weight_decay=self.preset.weight_decay,
                    lr_decay_after_epoch=self.preset.lr_decay_after_epoch,
                    lr_decay_factor=self.preset.lr_decay_factor,
                    restore_best=self.preset.restore_best,
                    debug_mode=os.getenv("CNN_DEBUG_MODE", "0").strip().lower() in {"1", "true", "yes", "on"},
                    debug_batch_size=max(1, int(os.getenv("CNN_DEBUG_BATCH_SIZE", "8"))),
                ),
            )
            pipeline.sanity_check(x_train[: self.preset.batch_size], y_train[: self.preset.batch_size])
            history = pipeline.train(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                on_epoch=lambda epoch, loss, acc, sec, hist: self.progress.emit(epoch, loss, acc, sec, hist.copy()),
            )
            cpu_model = pipeline.to_cpu_model()
            self.finished.emit(history, cpu_model)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class DrawingCanvas(QWidget):
    """280x280 drawing canvas for handwritten digit input."""

    changed = pyqtSignal()

    def __init__(self, size: int = 280) -> None:
        super().__init__()
        self.canvas_size = size
        self.setFixedSize(self.canvas_size, self.canvas_size)
        self._image = QImage(self.canvas_size, self.canvas_size, QImage.Format.Format_Grayscale8)
        self._image.fill(0)
        self._last_pos: QPoint | None = None
        self._brush_width = 18

    def paintEvent(self, event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.drawImage(self.rect(), self._image)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pos = event.position().toPoint()
            self._draw_point(self._last_pos)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if self._last_pos is not None:
                self._draw_line(self._last_pos, pos)
            self._last_pos = pos

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pos = None

    def clear(self) -> None:
        self._image.fill(0)
        self.update()
        self.changed.emit()

    def to_mnist_input(self) -> np.ndarray:
        """
        Convert canvas to MNIST-like tensor with shape (1, 1, 28, 28).
        Drawing is white-on-black, matching MNIST foreground/background convention.
        """
        ptr = self._image.constBits()
        raw = ptr.asstring(self.canvas_size * self.canvas_size)
        arr_280 = np.frombuffer(raw, dtype=np.uint8)
        arr_280 = arr_280.reshape(self.canvas_size, self.canvas_size)
        arr = arr_280.astype(np.float32) / 255.0
        # Improve robustness by centering/scaling the drawn foreground before 28x28 projection.
        arr_28 = self._to_mnist_centered(arr)
        return arr_28.reshape(1, 1, 28, 28)

    @staticmethod
    def _resize_nearest(src: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        in_h, in_w = src.shape
        y_idx = np.clip((np.arange(out_h) * (in_h / out_h)).astype(np.int32), 0, in_h - 1)
        x_idx = np.clip((np.arange(out_w) * (in_w / out_w)).astype(np.int32), 0, in_w - 1)
        return src[y_idx[:, None], x_idx[None, :]]

    def _to_mnist_centered(self, arr: np.ndarray) -> np.ndarray:
        mask = arr > 0.05
        if not np.any(mask):
            return np.zeros((28, 28), dtype=np.float32)

        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = arr[y0:y1, x0:x1]

        h, w = crop.shape
        target = 20  # common MNIST-like foreground size
        if h >= w:
            new_h = target
            new_w = max(1, int(round(w * (target / h))))
        else:
            new_w = target
            new_h = max(1, int(round(h * (target / w))))
        resized = self._resize_nearest(crop, new_h, new_w)

        canvas = np.zeros((28, 28), dtype=np.float32)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized

        max_v = float(canvas.max())
        if max_v > 0:
            canvas /= max_v
        return canvas

    def _draw_point(self, p: QPoint) -> None:
        self._draw_line(p, p)

    def _draw_line(self, p1: QPoint, p2: QPoint) -> None:
        painter = QPainter(self._image)
        pen = QPen(Qt.GlobalColor.white)
        pen.setWidth(self._brush_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()
        self.update()
        self.changed.emit()


class DrawingWindow(QWidget):
    """Separate window for live digit prediction from drawn input."""

    def __init__(self, model_getter: Callable[[], CNNModel | None]) -> None:
        super().__init__()
        self._model_getter = model_getter
        self.setWindowTitle("Draw Digit")
        self.resize(420, 520)

        self.canvas = DrawingCanvas(size=280)
        self.canvas.changed.connect(self.update_prediction)

        self.pred_label = QLabel("Prediction: -")

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.canvas.clear)

        self.prob_bars: list[QProgressBar] = []
        probs_layout = QFormLayout()
        for digit in range(10):
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
            self.prob_bars.append(bar)
            probs_layout.addRow(f"{digit}", bar)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignHCenter)
        root_layout.addWidget(self.clear_btn)
        root_layout.addWidget(self.pred_label)
        probs_widget = QWidget()
        probs_widget.setLayout(probs_layout)
        root_layout.addWidget(probs_widget)

    def update_prediction(self) -> None:
        model = self._model_getter()
        if model is None:
            self.pred_label.setText("Prediction: - (train or load model)")
            for bar in self.prob_bars:
                bar.setValue(0)
            return

        try:
            x = self.canvas.to_mnist_input()
            logits = model.forward(x, training=False)[0]
            shifted = logits - np.max(logits)
            exps = np.exp(shifted)
            probs = exps / np.sum(exps)

            pred = int(np.argmax(probs))
            self.pred_label.setText(f"Prediction: {pred}")
            for digit, bar in enumerate(self.prob_bars):
                bar.setValue(int(round(float(probs[digit]) * 100)))
        except Exception:
            self.pred_label.setText("Prediction: error")
            for bar in self.prob_bars:
                bar.setValue(0)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MNIST CNN Trainer (by Hepi34, ontaic07 and fritziii)")
        self.resize(920, 620)

        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._active_total_epochs: int = 1
        self.current_model: CNNModel | None = None
        self.current_model_path: Path | None = None
        self._active_preset_key: str | None = None
        self._active_device_key: str = "CPU"
        self._test_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._test_cache_paths: MNISTFilePaths | None = None
        self._drawing_window: DrawingWindow | None = None
        self.opencl_manager: OpenCLManager | None = OpenCLManager.create()
        self._gpu_probe_buffer: object | None = None

        default_dir = Path(__file__).resolve().parent.parent / "dataset" / "mnist-dataset"
        self.train_images_default = default_dir / "train-images.idx3.ubyte"
        self.train_labels_default = default_dir / "train-labels.idx1.ubyte"
        self.test_images_default = default_dir / "t10k-images.idx3.ubyte"
        self.test_labels_default = default_dir / "t10k-labels.idx1.ubyte"

        self.preset_combo = QComboBox()
        self._populate_preset_combo()
        self.preset_combo.currentTextChanged.connect(self._update_preset_details)
        self.preset_info_btn = QToolButton()
        self.preset_info_btn.setText("▸")
        self.preset_info_btn.setToolTip("Show preset details")
        self.preset_info_btn.clicked.connect(self._toggle_preset_details)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "GPU"])
        self.device_combo.setCurrentIndex(0)
        gpu_item = self.device_combo.model().item(1)
        if self.opencl_manager is None:
            self.device_combo.setItemText(1, "GPU (unavailable)")
            if gpu_item is not None:
                gpu_item.setEnabled(False)
        else:
            gpu_name = self.opencl_manager.info.device_name
            self.device_combo.setItemText(1, f"GPU ({gpu_name})")
            # Probe a tensor transfer to ensure buffer movement works.
            try:
                probe = np.zeros((1, 1, 28, 28), dtype=np.float32)
                self._gpu_probe_buffer = self.opencl_manager.to_device(probe)
            except Exception:
                self.opencl_manager = None
                self.device_combo.setItemText(1, "GPU (unavailable)")
                if gpu_item is not None:
                    gpu_item.setEnabled(False)

        self.thread_spin = QSpinBox()
        cpu_count = max(1, (os.cpu_count() or 1))
        self.thread_spin.setRange(1, cpu_count)
        self.thread_spin.setValue(min(4, cpu_count))

        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)

        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)

        self.eval_btn = QPushButton("Evaluate on Test Set")
        self.eval_btn.clicked.connect(self.evaluate_on_test_set)

        self.draw_btn = QPushButton("Open Drawing Window")
        self.draw_btn.clicked.connect(self.open_drawing_window)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.acc_label = QLabel("Accuracy: -")
        self.time_label = QLabel("Seconds per epoch: -")
        self.eta_label = QLabel("ETA: -")
        self.model_file_label = QLabel("Model file: -")
        self.preset_details_label = QLabel("")
        self.preset_details_label.setWordWrap(True)
        self.preset_details_label.setVisible(False)
        self._update_preset_details()
        self.train_images_edit = QLineEdit(str(self.train_images_default))
        self.train_labels_edit = QLineEdit(str(self.train_labels_default))
        self.test_images_edit = QLineEdit(str(self.test_images_default))
        self.test_labels_edit = QLineEdit(str(self.test_labels_default))

        controls = QWidget()
        form = QFormLayout(controls)
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.preset_info_btn)
        form.addRow("Preset", preset_row)
        form.addRow("", self.preset_details_label)
        form.addRow("Device", self.device_combo)
        form.addRow("CPU Threads", self.thread_spin)
        form.addRow("", self.start_btn)
        form.addRow("", self.load_btn)
        form.addRow("", self.eval_btn)
        form.addRow("", self.draw_btn)
        form.addRow("Progress", self.progress)
        form.addRow("", self.acc_label)
        form.addRow("", self.time_label)
        form.addRow("", self.eta_label)
        form.addRow("", self.model_file_label)
        form.addRow("Train Images", self._file_row(self.train_images_edit, self.pick_train_images))
        form.addRow("Train Labels", self._file_row(self.train_labels_edit, self.pick_train_labels))
        form.addRow("Test Images", self._file_row(self.test_images_edit, self.pick_test_images))
        form.addRow("Test Labels", self._file_row(self.test_labels_edit, self.pick_test_labels))

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax_loss = self.figure.add_subplot(111)
        self.ax_acc = self.ax_loss.twinx()
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_acc.set_ylabel("Accuracy")

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.addWidget(controls, stretch=0)
        layout.addWidget(self.canvas, stretch=1)
        self.setCentralWidget(root)

    def start_training(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return

        preset = self._current_preset()
        if preset is None:
            QMessageBox.information(self, "Select Preset", "Please select a concrete preset, not a version header.")
            return
        self._active_total_epochs = preset.epochs
        self._active_preset_key = preset.key
        self._active_device_key = "GPU" if self.device_combo.currentIndex() == 1 else "CPU"
        num_threads = int(self.thread_spin.value())
        paths = self._selected_paths()
        missing = [
            str(p)
            for p in (paths.train_images, paths.train_labels, paths.test_images, paths.test_labels)
            if not p.exists()
        ]
        if missing:
            QMessageBox.critical(self, "Dataset Missing", "Missing file(s):\n" + "\n".join(missing))
            return

        self.start_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.eval_btn.setEnabled(False)
        self.progress.setValue(0)
        self.acc_label.setText("Accuracy: -")
        self.time_label.setText("Seconds per epoch: -")
        self.eta_label.setText("ETA: -")
        self._draw_history({"loss": [], "accuracy": [], "epoch_time": []})

        self._thread = QThread()
        if self.device_combo.currentIndex() == 1:
            if self.opencl_manager is None:
                QMessageBox.critical(self, "GPU Unavailable", "No OpenCL GPU device is available.")
                self.start_btn.setEnabled(True)
                return
            self._worker = GPUTrainingWorker(
                preset=preset,
                paths=paths,
                opencl_manager=self.opencl_manager,
            )
        else:
            self._worker = TrainingWorker(preset=preset, num_threads=num_threads, paths=paths)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.finished.connect(self.on_finished)
        self._worker.failed.connect(self.on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def on_progress(self, epoch: int, loss: float, accuracy: float, sec: float, history: object) -> None:
        pct = int((epoch / max(1, self._active_total_epochs)) * 100)
        self.progress.setValue(pct)
        self.acc_label.setText(f"Accuracy: {accuracy * 100:.2f}%")
        self.time_label.setText(f"Seconds per epoch: {sec:.2f}s")
        epoch_times: list[float] = []
        if isinstance(history, dict):
            maybe_times = history.get("epoch_time", [])
            if isinstance(maybe_times, list):
                epoch_times = [float(t) for t in maybe_times]
        avg_epoch_sec = float(np.mean(epoch_times)) if epoch_times else float(sec)
        remaining_epochs = max(0, self._active_total_epochs - epoch)
        eta_sec = remaining_epochs * avg_epoch_sec
        self.eta_label.setText(f"ETA: {_format_eta_seconds(eta_sec)}")
        self._draw_history(history)

    def on_finished(self, history: object, model: object) -> None:
        self.start_btn.setEnabled(True)
        self.progress.setValue(100)
        self.eta_label.setText("ETA: 0s")
        self._draw_history(history)
        if isinstance(model, CNNModel):
            self.current_model = model
            preset = PRESETS.get(self._active_preset_key or "")
            version = preset.version if preset is not None else "vX"
            model_name = (preset.name if preset is not None else "Model").lower()
            device_name = self._active_device_key
            if device_name == "GPU":
                gpu_name = self.opencl_manager.info.device_name if self.opencl_manager is not None else "UnknownGPU"
                hardware_name = _sanitize_hardware_name(gpu_name)
            else:
                hardware_name = _sanitize_hardware_name(_detect_cpu_name())
            device_label = f"{device_name}_{hardware_name}"
            file_name = f"{version}{model_name}{device_label}.npz"
            default_save_dir = Path(__file__).resolve().parent.parent / "models"
            default_save_dir.mkdir(parents=True, exist_ok=True)
            default_save_path = default_save_dir / file_name
            try:
                self.current_model.save_weights(
                    default_save_path,
                    metadata={
                        "preset_key": self._active_preset_key or "",
                        "version": version,
                        "device": device_name,
                        "device_label": device_label,
                        "hardware_name": hardware_name,
                    },
                )
                self.current_model_path = default_save_path
                self.model_file_label.setText(f"Model file: {default_save_path}")
            except Exception:
                pass
        self.load_btn.setEnabled(True)
        self.eval_btn.setEnabled(True)

    def on_failed(self, message: str) -> None:
        self.start_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.eval_btn.setEnabled(True)
        self.eta_label.setText("ETA: -")
        QMessageBox.critical(self, "Training Error", message)
    
    def _on_thread_finished(self) -> None:
        self._worker = None
        self._thread = None

    def load_model(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Model Weights",
            str(Path(__file__).resolve().parent),
            "NumPy Weights (*.npz)",
        )
        if not file_path:
            return

        loaded_model: CNNModel | None = None
        loaded_preset_key: str | None = None
        errors: list[str] = []

        try:
            metadata = CNNModel.load_metadata(file_path)
        except Exception:
            metadata = {}
        preferred_key = metadata.get("preset_key", "").strip()
        inferred_version = ""
        match = re.search(r"(v\d+)", Path(file_path).stem.lower())
        if match is not None:
            inferred_version = match.group(1)

        ordered_keys: list[str] = []
        if preferred_key in PRESETS:
            ordered_keys.append(preferred_key)
        if inferred_version:
            ordered_keys.extend([k for k in PRESETS if k.startswith(f"{inferred_version}/") and k not in ordered_keys])
        ordered_keys.extend([k for k in PRESETS if k not in ordered_keys])

        for preset_key in ordered_keys:
            preset = PRESETS[preset_key]
            candidate = build_model(preset)
            try:
                candidate.load_weights(file_path)
                loaded_model = candidate
                loaded_preset_key = preset_key
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{preset_key}: {exc}")

        if loaded_model is None or loaded_preset_key is None:
            detail = "\n".join(errors)
            QMessageBox.critical(self, "Load Error", f"Could not load model file:\n{detail}")
            return

        self.current_model = loaded_model
        self.current_model_path = Path(file_path)
        self.preset_combo.setCurrentText(loaded_preset_key)
        self.model_file_label.setText(f"Model file: {file_path}")
        QMessageBox.information(
            self,
            "Model Loaded",
            f"Loaded weights from:\n{file_path}\nDetected preset: {loaded_preset_key}",
        )

    def evaluate_on_test_set(self) -> None:
        if self.current_model is None:
            QMessageBox.information(self, "No Model", "Train or load a model first.")
            return

        try:
            x_test, y_test = self._get_test_data()
            logits = self.current_model.forward(x_test, training=False)
            pred = np.argmax(logits, axis=1)
            accuracy = float(np.mean(pred == y_test))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Evaluation Error", str(exc))
            return

        self.acc_label.setText(f"Accuracy: {accuracy * 100:.2f}%")
        self._show_test_eval_table(y_true=y_test, y_pred=pred, total_accuracy=accuracy)

    def _get_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        selected = self._selected_paths()
        if self._test_cache is None or self._test_cache_paths != selected:
            _, _, x_test, y_test = load_mnist_from_files(
                train_images_path=selected.train_images,
                train_labels_path=selected.train_labels,
                test_images_path=selected.test_images,
                test_labels_path=selected.test_labels,
            )
            self._test_cache = (x_test, y_test)
            self._test_cache_paths = selected

        x_test, y_test = self._test_cache
        preset = self._current_preset()
        limit = preset.test_limit if preset is not None else min(10000, x_test.shape[0])
        return x_test[:limit], y_test[:limit]

    def open_drawing_window(self) -> None:
        if self._drawing_window is None:
            self._drawing_window = DrawingWindow(model_getter=lambda: self.current_model)
        self._drawing_window.show()
        self._drawing_window.raise_()
        self._drawing_window.activateWindow()

    def _draw_history(self, history_obj: object) -> None:
        history = history_obj if isinstance(history_obj, dict) else {}
        loss = history.get("loss", [])
        accuracy = history.get("accuracy", [])
        epochs = np.arange(1, len(loss) + 1)

        self.ax_loss.clear()
        self.ax_acc.clear()
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_acc.set_ylabel("Accuracy")

        if len(epochs) > 0:
            self.ax_loss.plot(epochs, loss, color="#1f77b4", marker="o", label="Loss")
            self.ax_acc.plot(epochs, accuracy, color="#d62728", marker="s", label="Accuracy")
            self.ax_acc.set_ylim(0.0, 1.0)

        self.canvas.draw_idle()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(5000)
        event.accept()

    def _show_test_eval_table(self, y_true: np.ndarray, y_pred: np.ndarray, total_accuracy: float) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Test Set Evaluation")
        dialog.resize(560, 420)

        layout = QVBoxLayout(dialog)
        total_label = QLabel(f"Total accuracy: {total_accuracy * 100:.2f}%")
        layout.addWidget(total_label)

        table = QTableWidget(10, 4, dialog)
        table.setHorizontalHeaderLabels(["Digit", "Present", "Correct", "Accuracy (%)"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        for digit in range(10):
            present = int(np.sum(y_true == digit))
            correct = int(np.sum((y_true == digit) & (y_pred == digit)))
            pct = (correct / present * 100.0) if present > 0 else 0.0

            table.setItem(digit, 0, QTableWidgetItem(str(digit)))
            table.setItem(digit, 1, QTableWidgetItem(str(present)))
            table.setItem(digit, 2, QTableWidgetItem(str(correct)))
            table.setItem(digit, 3, QTableWidgetItem(f"{pct:.2f}"))

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table)
        dialog.exec()

    def _selected_paths(self) -> MNISTFilePaths:
        return MNISTFilePaths(
            train_images=Path(self.train_images_edit.text()).expanduser(),
            train_labels=Path(self.train_labels_edit.text()).expanduser(),
            test_images=Path(self.test_images_edit.text()).expanduser(),
            test_labels=Path(self.test_labels_edit.text()).expanduser(),
        )

    def _file_row(self, edit: QLineEdit, picker: Callable[[], None]) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("Browse...")
        btn.clicked.connect(picker)
        layout.addWidget(edit)
        layout.addWidget(btn)
        return row

    def _pick_file(self, target: QLineEdit, title: str) -> None:
        start = target.text() or str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            start,
            "IDX/UByte Files (*.ubyte *.idx*);;All Files (*)",
        )
        if file_path:
            target.setText(file_path)
            self._test_cache = None
            self._test_cache_paths = None

    def pick_train_images(self) -> None:
        self._pick_file(self.train_images_edit, "Select Train Images File")

    def pick_train_labels(self) -> None:
        self._pick_file(self.train_labels_edit, "Select Train Labels File")

    def pick_test_images(self) -> None:
        self._pick_file(self.test_images_edit, "Select Test Images File")

    def pick_test_labels(self) -> None:
        self._pick_file(self.test_labels_edit, "Select Test Labels File")

    def _populate_preset_combo(self) -> None:
        self.preset_combo.clear()
        self.preset_combo.addItem("v1")
        self.preset_combo.addItem("v1/Mini")
        self.preset_combo.addItem("v1/Normal")
        self.preset_combo.addItem("v1/Pro")
        self.preset_combo.addItem("v2")
        self.preset_combo.addItem("v2/Mini")
        self.preset_combo.addItem("v2/Normal")
        self.preset_combo.addItem("v2/Pro")
        self.preset_combo.addItem("v3")
        self.preset_combo.addItem("v3/Mini")
        self.preset_combo.addItem("v3/Normal")
        self.preset_combo.addItem("v3/Pro")
        self.preset_combo.addItem("v3/Extreme")
        self.preset_combo.addItem("v4")
        self.preset_combo.addItem("v4/Mini")
        self.preset_combo.addItem("v4/Normal")
        self.preset_combo.addItem("v4/Pro")
        self.preset_combo.addItem("v4/Extreme")
        self.preset_combo.addItem("v5")
        self.preset_combo.addItem("v5/Mini")
        self.preset_combo.addItem("v5/Normal")
        self.preset_combo.addItem("v5/Pro")
        self.preset_combo.addItem("v5/Extreme")

        for idx in (0, 4, 8, 13, 18):
            item = self.preset_combo.model().item(idx)
            if item is not None:
                item.setEnabled(False)
        self.preset_combo.setCurrentText("v5/Normal")

    def _current_preset(self) -> Preset | None:
        key = self.preset_combo.currentText()
        return PRESETS.get(key)

    def _toggle_preset_details(self) -> None:
        visible = not self.preset_details_label.isVisible()
        self.preset_details_label.setVisible(visible)
        self.preset_info_btn.setText("▾" if visible else "▸")

    def _update_preset_details(self) -> None:
        preset = self._current_preset()
        if preset is None:
            self.preset_details_label.setText("Select a preset entry (e.g. v5/Normal).")
            return

        if preset.use_second_conv:
            c2 = int(preset.conv2_filters or (preset.conv_filters * 2))
            model_desc = (
                f"Conv2D(3x3,{preset.conv_filters}) -> ReLU -> Conv2D(3x3,{c2}) -> ReLU"
                + (" -> MaxPool2D(2x2)" if preset.use_maxpool else "")
                + f" -> Flatten -> Dense({preset.hidden_units}) -> ReLU -> Dense(10)"
            )
        else:
            model_desc = (
                f"Conv2D(3x3,{preset.conv_filters}) -> ReLU -> Flatten -> Dense({preset.hidden_units}) -> ReLU -> Dense(10)"
            )

        lines = [
            f"Version: {preset.version}",
            f"Model: {model_desc}",
            f"Optimizer: {preset.optimizer.upper()}",
            (f"Dropout: {preset.dropout_rate:.2f}" if preset.dropout_rate > 0.0 else "Dropout: off"),
            f"Epochs: {preset.epochs} | Batch: {preset.batch_size} | LR: {preset.lr}",
            (
                f"L2 weight decay: {preset.weight_decay}"
                + (
                    f" | LR decay: x{preset.lr_decay_factor} after epoch {preset.lr_decay_after_epoch}"
                    if preset.lr_decay_after_epoch is not None
                    else ""
                )
            ),
            f"Restore best epoch: {'on' if preset.restore_best else 'off'}",
            f"Train limit: {preset.train_limit} | Test limit: {preset.test_limit}",
        ]
        if preset.key in {"v4/Extreme", "v5/Extreme"}:
            lines.append("Note: highest-capacity preset; intended to push toward >=99% test accuracy.")
        self.preset_details_label.setText("\n".join(lines))


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

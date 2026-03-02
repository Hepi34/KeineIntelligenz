# cnn_backend.py
# CNN-Backend für Ziffern (0-9) mit Input-Größe 10x15 (cols x rows) => (15, 10, 1)

from __future__ import annotations

import os
import numpy as np

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -------------------------
# Daten laden (robust)
# -------------------------
def load_array(path: str) -> np.ndarray:
    """
    Lädt Arrays aus .npy oder .npz.
    - .npy: direktes Array
    - .npz: nimmt 'arr_0' oder (falls vorhanden) 'x'/'y'
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=False)

    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        # typische Keys probieren
        for key in ("arr_0", "x", "y", "images", "labels"):
            if key in data:
                return data[key]
        # falls irgendwas drin ist, nimm das erste
        keys = list(data.keys())
        if not keys:
            raise ValueError("Leere .npz-Datei.")
        return data[keys[0]]

    raise ValueError("Bitte .npy oder .npz verwenden.")


def ensure_image_shape(X: np.ndarray) -> np.ndarray:
    """
    Erwartet:
    - (N, 15, 10) oder (N, 15, 10, 1) oder (N, 10, 15) / (N, 10, 15, 1)
    Gibt zurück: (N, 15, 10, 1) float32 in [0..1]
    """
    X = np.asarray(X)

    if X.ndim == 3:
        # (N, H, W)
        pass
    elif X.ndim == 4:
        # (N, H, W, C)
        pass
    else:
        raise ValueError(f"Unerwartete Shape für Bilder: {X.shape}")

    # Kanal hinzufügen falls nötig
    if X.ndim == 3:
        X = X[..., np.newaxis]  # (N, H, W, 1)

    # jetzt (N, H, W, 1)
    if X.shape[1:4] == (15, 10, 1):
        pass
    elif X.shape[1:4] == (10, 15, 1):
        # tauschen (H,W)
        X = np.transpose(X, (0, 2, 1, 3))  # (N, 15, 10, 1)
    else:
        raise ValueError(
            f"Falsche Bildgröße. Erwartet (15,10,1) oder (10,15,1), bekommen: {X.shape[1:4]}"
        )

    # normalisieren
    X = X.astype("float32")
    # falls Werte 0..255 sind
    if X.max() > 1.5:
        X = X / 255.0

    # clamp
    X = np.clip(X, 0.0, 1.0)
    return X


def ensure_labels(y: np.ndarray) -> np.ndarray:
    """
    Erwartet y als:
    - (N,) ints 0..9
    - oder one-hot (N,10)
    Gibt zurück: (N,) int32
    """
    y = np.asarray(y)

    if y.ndim == 1:
        y_int = y.astype("int32")
    elif y.ndim == 2 and y.shape[1] == 10:
        y_int = np.argmax(y, axis=1).astype("int32")
    else:
        raise ValueError(f"Unerwartete Shape für Labels: {y.shape}")

    if y_int.min() < 0 or y_int.max() > 9:
        raise ValueError("Labels müssen zwischen 0 und 9 liegen.")
    return y_int


# -------------------------
# Model bauen
# -------------------------
def build_cnn_model(hidden_layers: int = 1, dense_units: int = 128, learning_rate: float = 1e-3) -> keras.Model:
    """
    hidden_layers = Anzahl Dense-Schichten (dein GUI-Feld "Hidden Layers")
    dense_units   = Neuronen pro Dense-Schicht
    Input: (15, 10, 1)
    """
    hidden_layers = int(hidden_layers)
    if hidden_layers < 0:
        hidden_layers = 0

    inputs = keras.Input(shape=(15, 10, 1))

    # Kleine CNN, passend für 10x15
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)

    # "Hidden Layers" als Dense-Stack
    for _ in range(hidden_layers):
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------
# Training (mit Callback für Progress)
# -------------------------
class _TkProgressCallback(keras.callbacks.Callback):
    def __init__(self, epochs: int, on_progress=None, on_status=None):
        super().__init__()
        self.epochs = max(1, int(epochs))
        self.on_progress = on_progress  # fn(percent:int)
        self.on_status = on_status      # fn(text:str)

    def on_train_begin(self, logs=None):
        if self.on_status:
            self.on_status("Training gestartet...")

    def on_epoch_end(self, epoch, logs=None):
        # epoch ist 0-basiert
        percent = int(((epoch + 1) / self.epochs) * 100)
        if self.on_progress:
            self.on_progress(percent)

        if self.on_status:
            acc = None
            if logs:
                acc = logs.get("accuracy", None)
            if acc is not None:
                self.on_status(f"Epoch {epoch+1}/{self.epochs} – acc: {acc:.3f}")
            else:
                self.on_status(f"Epoch {epoch+1}/{self.epochs}")

    def on_train_end(self, logs=None):
        if self.on_progress:
            self.on_progress(100)
        if self.on_status:
            self.on_status("Training fertig.")


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layers: int,
    epochs: int,
    batch_size: int = 64,
    validation_split: float = 0.1,
    on_progress=None,
    on_status=None,
) -> keras.Model:
    """
    Trainiert und gibt das trainierte Modell zurück.
    """
    Xp = ensure_image_shape(X)
    yp = ensure_labels(y)

    model = build_cnn_model(hidden_layers=hidden_layers)

    cb = _TkProgressCallback(epochs=epochs, on_progress=on_progress, on_status=on_status)

    model.fit(
        Xp, yp,
        epochs=int(epochs),
        batch_size=int(batch_size),
        validation_split=float(validation_split),
        shuffle=True,
        callbacks=[cb],
        verbose=0,  # GUI soll die Ausgabe steuern
    )
    return model


# -------------------------
# Speichern / Laden
# -------------------------
def save_model(model: keras.Model, path: str) -> None:
    """
    Empfehlung: *.keras (neues Format)
    """
    model.save(path)


def load_model(path: str) -> keras.Model:
    return keras.models.load_model(path)


# -------------------------
# Drawing -> Prediction
# -------------------------
def pixels_to_input(pixels_2d: list[list[int]] | np.ndarray) -> np.ndarray:
    """
    pixels_2d: 15x10 (rows x cols) mit 0/1
    Gibt zurück: (1, 15, 10, 1) float32
    """
    arr = np.array(pixels_2d, dtype="float32")  # (15,10)
    if arr.shape != (15, 10):
        raise ValueError(f"Erwartet 15x10 Pixels, bekommen: {arr.shape}")

    arr = np.clip(arr, 0.0, 1.0)
    arr = arr[np.newaxis, ..., np.newaxis]  # (1,15,10,1)
    return arr


def predict_digit(model: keras.Model, pixels_2d) -> tuple[int, float, np.ndarray]:
    """
    Returns: (digit, certainty_percent, probs[10])
    """
    x = pixels_to_input(pixels_2d)
    probs = model.predict(x, verbose=0)[0]  # (10,)
    digit = int(np.argmax(probs))
    certainty = float(probs[digit] * 100.0)
    return digit, certainty, probs
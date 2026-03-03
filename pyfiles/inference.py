"""
inference.py — Model loading and prediction for both CPU and GPU trained models.

Handles two pkl formats:
  - CPU model: a raw pickled SimpleCNN object (has .forward() directly)
  - GPU model: a dict with key '_gpu_model': True and numpy weight arrays
               (weights are loaded into a CPU SimpleCNN for inference)
"""

import numpy as np
import pickle


# ---------------------------------------------------------------------------
# Internal: rebuild a CPU SimpleCNN from a GPU weight dict
# ---------------------------------------------------------------------------

def _load_gpu_weights_into_cpu(state: dict):
    """Reconstruct a SimpleCNN from the numpy weight dict saved by train_gpu_cnn."""
    from train_cnn_cpu import SimpleCNN

    num_filters = state['num_filters']
    # Infer hidden_size from fc1 weight shape
    hidden_size = state['fc1_W'].shape[1]

    model = SimpleCNN(num_filters=num_filters, hidden_layers=hidden_size)

    model.conv1.filters = state['conv1_filters'].astype(np.float32)
    model.conv1.biases  = state['conv1_biases'].astype(np.float32)
    model.conv2.filters = state['conv2_filters'].astype(np.float32)
    model.conv2.biases  = state['conv2_biases'].astype(np.float32)
    model.fc1.W = state['fc1_W'].astype(np.float32)
    model.fc1.b = state['fc1_b'].astype(np.float32)
    model.fc2.W = state['fc2_W'].astype(np.float32)
    model.fc2.b = state['fc2_b'].astype(np.float32)

    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(filepath: str):
    """
    Load a model from a pkl file.

    Returns a SimpleCNN object ready for inference regardless of whether
    it was originally trained on CPU or GPU.
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and obj.get('_gpu_model'):
        model = _load_gpu_weights_into_cpu(obj)
    else:
        # Raw CPU SimpleCNN
        model = obj

    return model


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict(model, image: np.ndarray):
    """
    Predict a single image or a batch.

    Args:
        model:  SimpleCNN returned by load_model()
        image:  numpy array, any of:
                  - (H, W)         single greyscale image
                  - (1, H, W)      single image with channel dim
                  - (1, 1, H, W)   single image in batch/channel format
                  - (N, 1, H, W)   batch

    Returns:
        predicted_class (int or 1-D array of ints),
        confidence      (float or 1-D array of floats, 0–100 %)
    """
    img = np.array(image, dtype=np.float32)

    # Normalise to [0, 1] if needed
    if img.max() > 1.0:
        img = img / 255.0

    # Ensure shape (N, 1, H, W)
    if img.ndim == 2:                   # (H, W)
        img = img[np.newaxis, np.newaxis, :, :]
    elif img.ndim == 3 and img.shape[0] == 1:   # (1, H, W)
        img = img[np.newaxis, :, :, :]
    elif img.ndim == 3:                 # (H, W, C) — shouldn't happen but guard
        img = img[:, :, 0][np.newaxis, np.newaxis, :, :]
    # else already (N, 1, H, W)

    # Resize to 28×28 if needed (e.g. drawing area is 15×10)
    H, W = img.shape[2], img.shape[3]
    if H != 28 or W != 28:
        img = _resize_batch(img, 28, 28)

    logits = model.forward(img)
    probs  = softmax(logits)

    classes     = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(probs)), classes] * 100.0

    if len(classes) == 1:
        return int(classes[0]), float(confidences[0])
    return classes, confidences


def evaluate(model, images: np.ndarray, labels: np.ndarray,
             batch_size: int = 128) -> dict:
    """
    Evaluate model accuracy on a labelled dataset.

    Args:
        model:      SimpleCNN returned by load_model()
        images:     numpy array (N, H, W) or (N, 1, H, W), uint8 or float32
        labels:     numpy array (N,) of integer class indices
        batch_size: how many images to process at once

    Returns:
        dict with keys:
          'accuracy'     – float, 0–100 %
          'correct'      – int
          'total'        – int
          'per_class'    – dict {class_int: {'correct': int, 'total': int}}
    """
    imgs = np.array(images, dtype=np.float32)
    if imgs.max() > 1.0:
        imgs /= 255.0

    if imgs.ndim == 3:                  # (N, H, W) → (N, 1, H, W)
        imgs = imgs[:, np.newaxis, :, :]

    H, W = imgs.shape[2], imgs.shape[3]
    if H != 28 or W != 28:
        imgs = _resize_batch(imgs, 28, 28)

    lbls = np.array(labels, dtype=np.int32)
    n = len(lbls)

    all_preds = []
    for i in range(0, n, batch_size):
        batch = imgs[i:i + batch_size]
        logits = model.forward(batch)
        preds  = np.argmax(logits, axis=1)
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds)
    correct   = int((all_preds == lbls).sum())

    per_class = {}
    for c in range(10):
        mask = (lbls == c)
        per_class[c] = {
            'correct': int((all_preds[mask] == c).sum()),
            'total':   int(mask.sum()),
        }

    return {
        'accuracy':  correct / n * 100.0,
        'correct':   correct,
        'total':     n,
        'per_class': per_class,
    }


# ---------------------------------------------------------------------------
# Internal: nearest-neighbour resize (no scipy/PIL dependency)
# ---------------------------------------------------------------------------

def _resize_batch(imgs: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a batch (N, C, H, W) to (N, C, out_h, out_w) via bilinear interp."""
    N, C, H, W = imgs.shape
    row_idx = (np.arange(out_h) * H / out_h).astype(np.float32)
    col_idx = (np.arange(out_w) * W / out_w).astype(np.float32)

    r0 = np.floor(row_idx).astype(int).clip(0, H - 1)
    r1 = (r0 + 1).clip(0, H - 1)
    c0 = np.floor(col_idx).astype(int).clip(0, W - 1)
    c1 = (c0 + 1).clip(0, W - 1)

    dr = (row_idx - r0)[:, np.newaxis]   # (out_h, 1)
    dc = (col_idx - c0)[np.newaxis, :]   # (1, out_w)

    out = np.zeros((N, C, out_h, out_w), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            img = imgs[n, c]
            out[n, c] = (img[r0][:, c0] * (1 - dr) * (1 - dc) +
                         img[r0][:, c1] * (1 - dr) * dc +
                         img[r1][:, c0] * dr        * (1 - dc) +
                         img[r1][:, c1] * dr        * dc)
    return out

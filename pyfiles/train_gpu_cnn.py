"""
train_gpu_cnn.py — GPU-accelerated CNN training without PyTorch or TensorFlow.

Backends:
  • CUDA  → CuPy   (pip install cupy-cuda12x)   – drop-in NumPy for NVIDIA GPUs
  • Metal → MLX    (pip install mlx)             – Apple's array framework for M-series Macs

Both libraries expose a NumPy-compatible API, so the same layer code works for both.
The `xp` variable is the active array module (cupy or mlx.core), chosen at startup.
"""

import numpy as np
import pickle


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def _select_backend(gpu_type: str):
    """Return the array module for the requested GPU backend."""
    if gpu_type == 'cuda':
        try:
            import cupy as cp
            cp.array([1])  # smoke-test device access
            print(f"GPU backend: CuPy (CUDA) — device {cp.cuda.Device().id}")
            return cp
        except Exception as e:
            raise RuntimeError(
                f"CUDA requested but CuPy initialisation failed: {e}\n"
                "Install with: pip install cupy-cuda12x  (match your CUDA version)"
            )

    elif gpu_type == 'metal':
        try:
            import mlx.core as mx
            mx.array([1])
            print("GPU backend: MLX (Apple Metal)")
            return mx
        except Exception as e:
            raise RuntimeError(
                f"Metal requested but MLX initialisation failed: {e}\n"
                "Install with: pip install mlx"
            )

    else:
        raise ValueError(f"Unknown gpu_type '{gpu_type}'. Use 'cuda' or 'metal'.")


# ---------------------------------------------------------------------------
# Utility: move numpy arrays to GPU, move results back to numpy
# ---------------------------------------------------------------------------

def to_gpu(arr, xp):
    """Send a numpy array to the GPU array module."""
    if hasattr(xp, 'asarray'):
        return xp.asarray(arr)
    return xp.array(arr)


def to_numpy(arr, xp):
    """Bring a GPU array back to numpy."""
    if hasattr(arr, 'get'):          # CuPy
        return arr.get()
    if hasattr(xp, 'eval'):          # MLX: materialise lazy graph first
        xp.eval(arr)
        return np.array(arr)
    return np.array(arr)


# ---------------------------------------------------------------------------
# Layers — written against xp so they run on CUDA or Metal transparently
# ---------------------------------------------------------------------------

class ConvLayerGPU:
    """3×3 same-padding convolutional layer."""

    def __init__(self, num_filters, input_depth, learning_rate, xp):
        self.xp = xp
        self.num_filters = num_filters
        self.input_depth = input_depth
        self.lr = learning_rate

        fan_in = input_depth * 9
        scale = float(np.sqrt(2.0 / fan_in))
        self.filters = xp.asarray(
            np.random.randn(num_filters, input_depth, 3, 3).astype(np.float32) * scale
        )
        self.biases = xp.zeros(num_filters, dtype=xp.float32)

    def forward(self, X):
        xp = self.xp
        self.X = X
        batch_size, depth, H, W = X.shape

        X_pad = xp.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)))
        out = xp.zeros((batch_size, self.num_filters, H, W), dtype=xp.float32)
        f_flat = self.filters.reshape(self.num_filters, -1)   # (F, D*9)

        for i in range(H):
            for j in range(W):
                patch = X_pad[:, :, i:i+3, j:j+3].reshape(batch_size, -1)   # (B, D*9)
                out[:, :, i, j] = patch @ f_flat.T + self.biases             # (B, F)

        return out

    def backward(self, dout):
        xp = self.xp
        batch_size, _, H, W = dout.shape
        X_pad = xp.pad(self.X, ((0, 0), (0, 0), (1, 1), (1, 1)))
        dX_pad = xp.zeros_like(X_pad)
        df = xp.zeros_like(self.filters)
        db = dout.sum(axis=(0, 2, 3))
        f_flat = self.filters.reshape(self.num_filters, -1)

        for i in range(H):
            for j in range(W):
                patch = X_pad[:, :, i:i+3, j:j+3].reshape(batch_size, -1)   # (B, D*9)
                d_ij = dout[:, :, i, j]                                       # (B, F)
                df += (d_ij.T @ patch).reshape(self.filters.shape)
                dX_patch = (d_ij @ f_flat).reshape(batch_size, self.input_depth, 3, 3)
                dX_pad[:, :, i:i+3, j:j+3] += dX_patch

        self.filters -= self.lr * df / batch_size
        self.biases  -= self.lr * db / batch_size
        return dX_pad[:, :, 1:-1, 1:-1]


class PoolLayerGPU:
    """2×2 max-pool, stride 2."""

    def __init__(self, xp):
        self.xp = xp

    def forward(self, X):
        xp = self.xp
        self.X = X
        B, D, H, W = X.shape
        oh, ow = H // 2, W // 2
        Xr = X[:, :, :oh*2, :ow*2].reshape(B, D, oh, 2, ow, 2)
        return Xr.max(axis=(3, 5))

    def backward(self, dout):
        xp = self.xp
        B, D, H, W = self.X.shape
        oh, ow = H // 2, W // 2
        dX = xp.zeros_like(self.X)
        Xr = self.X[:, :, :oh*2, :ow*2].reshape(B, D, oh, 2, ow, 2)
        max_v = Xr.max(axis=(3, 5), keepdims=True)
        mask = (Xr == max_v).astype(xp.float32)
        mask /= xp.maximum(mask.sum(axis=(3, 5), keepdims=True), xp.array(1.0))
        # Use reshape instead of np.newaxis — works on both CuPy and MLX
        dout_exp = dout.reshape(B, D, oh, 1, ow, 1)
        dX[:, :, :oh*2, :ow*2] = (mask * dout_exp).reshape(B, D, oh*2, ow*2)
        return dX


class DenseLayerGPU:
    """Fully-connected layer."""

    def __init__(self, in_size, out_size, learning_rate, xp):
        self.xp = xp
        self.lr = learning_rate
        scale = float(np.sqrt(2.0 / in_size))
        self.W = xp.asarray(np.random.randn(in_size, out_size).astype(np.float32) * scale)
        self.b = xp.zeros(out_size, dtype=xp.float32)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout):
        dX = dout @ self.W.T
        dW = self.X.T @ dout
        db = dout.sum(axis=0)
        self.W -= self.lr * dW / self.X.shape[0]
        self.b  -= self.lr * db / self.X.shape[0]
        return dX


# ---------------------------------------------------------------------------
# Full CNN model
# ---------------------------------------------------------------------------

class SimpleCNNGPU:
    def __init__(self, num_filters=16, hidden_size=128, learning_rate=0.001, xp=None, gpu_type=None):
        assert xp is not None, "Pass xp (cupy or mlx.core)"
        self.xp = xp
        self.gpu_type = gpu_type  # string, safe to pickle
        self.num_filters = num_filters

        self.conv1 = ConvLayerGPU(num_filters,       1,             learning_rate, xp)
        self.pool1 = PoolLayerGPU(xp)
        self.conv2 = ConvLayerGPU(num_filters * 2,   num_filters,   learning_rate, xp)
        self.pool2 = PoolLayerGPU(xp)

        flat = num_filters * 2 * 7 * 7  # 28×28 input → two 2×2 pools → 7×7
        self.fc1 = DenseLayerGPU(flat,      hidden_size, learning_rate, xp)
        self.fc2 = DenseLayerGPU(hidden_size, 10,        learning_rate, xp)

    def forward(self, X):
        xp = self.xp
        self.bs = X.shape[0]

        X = self.conv1.forward(X);  self.r1 = X = xp.maximum(X, 0)
        X = self.pool1.forward(X)
        X = self.conv2.forward(X);  self.r2 = X = xp.maximum(X, 0)
        X = self.pool2.forward(X)

        X = X.reshape(self.bs, -1)
        X = self.fc1.forward(X);    self.r3 = X = xp.maximum(X, 0)
        X = self.fc2.forward(X)
        return X

    def backward(self, dout):
        dout = self.fc2.backward(dout)
        dout = dout * (self.r3 > 0)
        dout = self.fc1.backward(dout)
        dout = dout.reshape(self.bs, self.num_filters * 2, 7, 7)
        dout = self.pool2.backward(dout)
        dout = dout * (self.r2 > 0)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = dout * (self.r1 > 0)
        self.conv1.backward(dout)

    def __getstate__(self):
        raise TypeError(
            "SimpleCNNGPU cannot be pickled directly because it holds GPU array "
            "references. Use train_gpu_cnn.save_model() to save the model."
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def softmax_gpu(x, xp):
    e = xp.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_gpu(logits, labels, xp):
    probs = softmax_gpu(logits, xp)
    bs = labels.shape[0]
    # index log-probabilities at correct class
    log_p = xp.log(probs[xp.arange(bs), labels] + 1e-8)
    loss = -log_p.mean()
    return loss, probs


# ---------------------------------------------------------------------------
# Data loading (identical to CPU version, returns numpy; we move to GPU later)
# ---------------------------------------------------------------------------

def load_ubyte_data(data_file, label_file):
    with open(data_file, 'rb') as f:
        np.frombuffer(f.read(4), dtype='>i4')   # magic
        n     = int(np.frombuffer(f.read(4), dtype='>i4')[0])
        rows  = int(np.frombuffer(f.read(4), dtype='>i4')[0])
        cols  = int(np.frombuffer(f.read(4), dtype='>i4')[0])
        imgs  = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    with open(label_file, 'rb') as f:
        np.frombuffer(f.read(4), dtype='>i4')   # magic
        np.frombuffer(f.read(4), dtype='>i4')   # count
        lbls = np.frombuffer(f.read(), dtype=np.uint8)

    return imgs, lbls


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_cnn(dataset_path, labels_path, epochs, hidden_layers,
              gpu_type='cuda', callback=None):
    """
    Train CNN on GPU.

    Args:
        dataset_path:  Path to ubyte images file
        labels_path:   Path to ubyte labels file
        epochs:        Training epochs
        hidden_layers: FC hidden-layer size
        gpu_type:      'cuda' or 'metal'
        callback:      Optional fn(current_epoch, total_epochs, avg_loss)

    Returns:
        Trained SimpleCNNGPU model  (weights live on the GPU device)
    """
    xp = _select_backend(gpu_type)

    print(f"Loading data...")
    imgs_np, lbls_np = load_ubyte_data(dataset_path, labels_path)

    imgs_np = imgs_np.astype(np.float32) / 255.0
    imgs_np = imgs_np[:, np.newaxis, :, :]   # (N,1,H,W)

    # Move entire dataset to GPU at once (fast for MNIST-sized data)
    imgs_gpu = to_gpu(imgs_np, xp)
    lbls_gpu = to_gpu(lbls_np.astype(np.int32), xp)

    n = imgs_np.shape[0]
    batch_size = 128   # larger batch exploits GPU parallelism better
    model = SimpleCNNGPU(num_filters=16, hidden_size=hidden_layers,
                         learning_rate=0.001, xp=xp, gpu_type=gpu_type)

    print(f"Training {epochs} epochs on {gpu_type.upper()} GPU...")

    for epoch in range(epochs):
        perm = xp.asarray(np.random.permutation(n))
        imgs_s = imgs_gpu[perm]
        lbls_s = lbls_gpu[perm]

        total_loss = 0.0
        nb = 0

        for i in range(0, n, batch_size):
            bx = imgs_s[i:i+batch_size]
            by = lbls_s[i:i+batch_size]

            logits = model.forward(bx)
            loss, probs = cross_entropy_gpu(logits, by, xp)

            # Build one-hot gradient: MLX has no indexed assignment or .copy()
            # xp.eye gives a clean (10,10) identity; indexing by labels gives (B,10) one-hot
            bs_actual = int(bx.shape[0])
            one_hot = xp.array(np.eye(10, dtype=np.float32))[by]   # (B, 10)
            dout = (probs - one_hot) / bs_actual
            model.backward(dout)

            # Materialise scalar loss for Python (triggers MLX graph eval)
            loss_val = float(to_numpy(loss, xp))
            total_loss += loss_val
            nb += 1

        avg = total_loss / nb
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}")
        if callback:
            callback(epoch + 1, epochs, avg)

    return model


# ---------------------------------------------------------------------------
# Persistence  — weights transferred back to numpy so pickle works anywhere
# ---------------------------------------------------------------------------

def save_model(model, filepath):
    """Save GPU model to disk.

    Converts all GPU arrays to plain numpy arrays before pickling so the
    file can be loaded on any machine regardless of GPU availability.
    The module reference (xp) is never pickled — only gpu_type string is kept.
    """
    xp = model.xp

    def _np(arr):
        return to_numpy(arr, xp)

    # Build a pure-numpy state dict — no GPU arrays, no module references
    state = {
        '_gpu_model': True,
        'gpu_type':   model.gpu_type,
        'num_filters': model.num_filters,
        'conv1_filters': _np(model.conv1.filters),
        'conv1_biases':  _np(model.conv1.biases),
        'conv2_filters': _np(model.conv2.filters),
        'conv2_biases':  _np(model.conv2.biases),
        'fc1_W': _np(model.fc1.W), 'fc1_b': _np(model.fc1.b),
        'fc2_W': _np(model.fc2.W), 'fc2_b': _np(model.fc2.b),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f"GPU model saved to {filepath}")


if __name__ == "__main__":
    import sys
    gpu = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    model = train_cnn(
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
        epochs=2,
        hidden_layers=128,
        gpu_type=gpu
    )
    save_model(model, "cnn_model_gpu.pkl")

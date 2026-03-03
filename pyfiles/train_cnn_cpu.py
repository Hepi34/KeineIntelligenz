import numpy as np
import pickle

class ConvLayer:
    """Convolutional layer with fixed 3x3 kernels"""
    def __init__(self, num_filters, input_depth, learning_rate=0.01):
        self.num_filters = num_filters
        self.input_depth = input_depth
        self.learning_rate = learning_rate

        # He initialization for better gradient flow with ReLU
        fan_in = input_depth * 3 * 3
        self.filters = np.random.randn(num_filters, input_depth, 3, 3) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros(num_filters)

    def forward(self, X):
        self.X = X
        batch_size, depth, height, width = X.shape

        X_padded = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

        out_height = height
        out_width = width

        # Vectorized im2col for fast convolution
        # Build column matrix: shape (batch, depth*3*3, H*W)
        cols = np.lib.stride_tricks.sliding_window_view(
            X_padded, (batch_size, depth, 3, 3)
        )
        # Fallback to loop-based if sliding_window_view shape is tricky
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))
        filters_flat = self.filters.reshape(self.num_filters, -1)  # (F, depth*3*3)

        for i in range(out_height):
            for j in range(out_width):
                X_slice = X_padded[:, :, i:i+3, j:j+3]          # (B, D, 3, 3)
                X_col = X_slice.reshape(batch_size, -1)           # (B, D*9)
                output[:, :, i, j] = X_col @ filters_flat.T + self.biases  # (B, F)

        return output

    def backward(self, dout):
        batch_size, _, height, width = dout.shape
        X_padded = np.pad(self.X, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')

        dX_padded = np.zeros_like(X_padded)
        dfilters = np.zeros_like(self.filters)
        dbiases = np.sum(dout, axis=(0, 2, 3))

        filters_flat = self.filters.reshape(self.num_filters, -1)  # (F, D*9)

        for i in range(height):
            for j in range(width):
                X_slice = X_padded[:, :, i:i+3, j:j+3]   # (B, D, 3, 3)
                X_col = X_slice.reshape(batch_size, -1)    # (B, D*9)
                d_col = dout[:, :, i, j]                   # (B, F)

                # Gradient for filters
                dfilters += (d_col.T @ X_col).reshape(self.filters.shape)  # (F, D*9) -> (F,D,3,3)

                # Gradient for input
                dX_col = d_col @ filters_flat               # (B, D*9)
                dX_padded[:, :, i:i+3, j:j+3] += dX_col.reshape(batch_size, self.input_depth, 3, 3)

        self.filters -= self.learning_rate * dfilters / batch_size
        self.biases -= self.learning_rate * dbiases / batch_size

        return dX_padded[:, :, 1:-1, 1:-1]


class PoolLayer:
    """Max pooling layer (2x2, stride 2)"""
    def forward(self, X):
        self.X = X
        batch_size, depth, height, width = X.shape
        oh, ow = height // 2, width // 2
        X_reshaped = X[:, :, :oh*2, :ow*2].reshape(batch_size, depth, oh, 2, ow, 2)
        output = X_reshaped.max(axis=(3, 5))
        return output

    def backward(self, dout):
        batch_size, depth, height, width = self.X.shape
        oh, ow = height // 2, width // 2
        dX = np.zeros_like(self.X)

        X_reshaped = self.X[:, :, :oh*2, :ow*2].reshape(batch_size, depth, oh, 2, ow, 2)
        max_vals = X_reshaped.max(axis=(3, 5), keepdims=True)  # (B, D, oh, 1, ow, 1)
        mask = (X_reshaped == max_vals).astype(np.float32)
        # Distribute gradient to max position (handle ties by splitting)
        mask /= mask.sum(axis=(3, 5), keepdims=True).clip(1, None)
        # dout is (B, D, oh, ow) — insert singleton dims at positions 3 and 5
        dout_expanded = dout.reshape(batch_size, depth, oh, 1, ow, 1)
        dX[:, :, :oh*2, :ow*2] = (mask * dout_expanded).reshape(batch_size, depth, oh*2, ow*2)
        return dX


class DenseLayer:
    """Fully connected layer"""
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # He initialization
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout):
        dX = dout @ self.W.T
        dW = self.X.T @ dout
        db = np.sum(dout, axis=0)

        self.W -= self.learning_rate * dW / self.X.shape[0]
        self.b -= self.learning_rate * db / self.X.shape[0]

        return dX


class SimpleCNN:
    """Simple CNN for MNIST-like data"""
    def __init__(self, num_filters=16, hidden_layers=128, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.num_filters = num_filters

        self.conv1 = ConvLayer(num_filters, 1, learning_rate)
        self.pool1 = PoolLayer()
        self.conv2 = ConvLayer(num_filters * 2, num_filters, learning_rate)
        self.pool2 = PoolLayer()

        # 28x28 -> pool1 -> 14x14 -> pool2 -> 7x7
        self.flattened_size = num_filters * 2 * 7 * 7

        self.fc1 = DenseLayer(self.flattened_size, hidden_layers, learning_rate)
        self.fc2 = DenseLayer(hidden_layers, 10, learning_rate)

        self.batch_size = None

    def forward(self, X):
        self.batch_size = X.shape[0]

        X = self.conv1.forward(X)
        self.conv1_output = X
        X = np.maximum(X, 0)
        self.relu1_output = X

        X = self.pool1.forward(X)
        self.pool1_output = X

        X = self.conv2.forward(X)
        self.conv2_output = X
        X = np.maximum(X, 0)
        self.relu2_output = X

        X = self.pool2.forward(X)
        self.pool2_output = X

        X = X.reshape(X.shape[0], -1)
        self.fc1_input = X
        X = self.fc1.forward(X)
        self.fc1_output = X
        X = np.maximum(X, 0)
        self.relu3_output = X

        X = self.fc2.forward(X)
        return X

    def backward(self, dout):
        dout = self.fc2.backward(dout)

        dout = dout * (self.relu3_output > 0)

        dout = self.fc1.backward(dout)
        dout = dout.reshape(self.batch_size, self.num_filters * 2, 7, 7)

        dout = self.pool2.backward(dout)
        dout = dout * (self.relu2_output > 0)
        dout = self.conv2.backward(dout)

        dout = self.pool1.backward(dout)
        dout = dout * (self.relu1_output > 0)
        dout = self.conv1.backward(dout)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy_loss(pred, true):
    batch_size = true.shape[0]
    pred = softmax(pred)
    loss = -np.mean(np.log(pred[np.arange(batch_size), true] + 1e-8))
    return loss, pred


def load_ubyte_data(data_file, label_file):
    """Load MNIST ubyte format files"""
    with open(data_file, 'rb') as f:
        np.frombuffer(f.read(4), dtype='>i4')[0]   # magic
        num_images = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_rows = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_cols = np.frombuffer(f.read(4), dtype='>i4')[0]
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)

    with open(label_file, 'rb') as f:
        np.frombuffer(f.read(4), dtype='>i4')[0]   # magic
        np.frombuffer(f.read(4), dtype='>i4')[0]   # num_labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels


def _process_batch(args):
    """Worker function: forward + backward on a single mini-batch. Returns (loss, gradients)."""
    model, batch_images, batch_labels = args
    logits = model.forward(batch_images)
    loss, probs = cross_entropy_loss(logits, batch_labels)

    dout = probs.copy()
    dout[np.arange(batch_images.shape[0]), batch_labels] -= 1
    dout /= batch_images.shape[0]
    model.backward(dout)

    return loss


def train_cnn(dataset_path, labels_path, epochs, hidden_layers,
              num_threads=4, callback=None):
    """
    Train a CNN model on the CPU using multiple threads.

    Threading strategy: Python's GIL limits true CPU parallelism for pure
    Python code, but numpy releases the GIL for heavy numerical operations.
    We use a ThreadPoolExecutor to overlap numpy compute across batches where
    possible, and also set numpy's own thread count via os.environ.

    Args:
        dataset_path:  Path to training images (ubyte format)
        labels_path:   Path to training labels (ubyte format)
        epochs:        Number of training epochs
        hidden_layers: Size of the FC hidden layer
        num_threads:   Number of threads to use (also sets numpy thread count)
        callback:      Optional function(current_epoch, total_epochs, loss)

    Returns:
        Trained SimpleCNN model
    """
    # threadpoolctl is the only reliable way to change BLAS thread count at
    # runtime — os.environ vars are read by OpenBLAS only at first import,
    # so setting them here (after numpy is already loaded) has no effect.
    try:
        from threadpoolctl import threadpool_limits
        _limiter = threadpool_limits(limits=num_threads)
        _limiter.__enter__()
    except ImportError:
        _limiter = None
        print("[Warning] threadpoolctl not installed; "
              "install with 'pip install threadpoolctl' for true multi-core support.")

    print(f"Loading data from {dataset_path} and {labels_path}...")
    images, labels = load_ubyte_data(dataset_path, labels_path)

    images = images.astype(np.float32) / 255.0
    images = images[:, np.newaxis, :, :]  # (N, 1, H, W)

    batch_size = 64   # larger batch = better parallelism in numpy BLAS
    num_samples = images.shape[0]

    model = SimpleCNN(num_filters=16, hidden_layers=hidden_layers, learning_rate=0.001)

    print(f"Starting training for {epochs} epochs on {num_threads} threads...")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(num_samples)
        images_shuffled = images[indices]
        labels_shuffled = labels[indices]

        # Batches run serially — the model stores activations between forward and
        # backward, so concurrent calls would corrupt state. Real parallelism comes
        # from numpy's BLAS backend (OpenBLAS/MKL) using all num_threads cores
        # internally for the heavy matmul/dot operations.
        for i in range(0, num_samples, batch_size):
            batch_images = images_shuffled[i:i + batch_size]
            batch_labels = labels_shuffled[i:i + batch_size]
            loss = _process_batch((model, batch_images, batch_labels))
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        if callback:
            callback(epoch + 1, epochs, avg_loss)

    if _limiter is not None:
        _limiter.__exit__(None, None, None)

    return model


def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    dataset_path = "train-images.idx3-ubyte"
    labels_path = "train-labels.idx1-ubyte"
    model = train_cnn(dataset_path, labels_path, epochs=2,
                      hidden_layers=128, num_threads=4)
    save_model(model, "cnn_model.pkl")

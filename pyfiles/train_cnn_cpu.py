import numpy as np
import pickle
import os


class ConvLayer:
    """Convolutional layer with fixed 3x3 kernels"""
    def __init__(self, num_filters, input_depth, learning_rate=0.01):
        self.num_filters = num_filters
        self.input_depth = input_depth
        self.learning_rate = learning_rate
        
        # Initialize filters
        self.filters = np.random.randn(num_filters, input_depth, 3, 3) * 0.01
        self.biases = np.zeros(num_filters)
        
    def forward(self, X):
        """Forward pass with padding"""
        self.X = X
        batch_size, depth, height, width = X.shape
        
        # Pad input
        X_padded = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
        
        # Output dimensions
        out_height = height
        out_width = width
        output = np.zeros((batch_size, self.num_filters, out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                X_slice = X_padded[:, :, i:i+3, j:j+3]
                for f in range(self.num_filters):
                    output[:, f, i, j] = np.sum(X_slice * self.filters[f], axis=(1, 2, 3)) + self.biases[f]
        
        return output
    
    def backward(self, dout):
        """Backward pass"""
        batch_size, _, height, width = dout.shape
        X_padded = np.pad(self.X, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
        
        dX = np.zeros_like(X_padded)
        dfilters = np.zeros_like(self.filters)
        dbiases = np.sum(dout, axis=(0, 2, 3))
        
        for i in range(height):
            for j in range(width):
                X_slice = X_padded[:, :, i:i+3, j:j+3]
                for f in range(self.num_filters):
                    # Gradient for filters: X_slice is (batch, input_depth, 3, 3)
                    # dout[:, f, i, j] is (batch,)
                    grad = (X_slice * dout[:, f, i, j][:, np.newaxis, np.newaxis, np.newaxis]).sum(axis=0)                    # Gradient for input

                    dX[:, :, i:i+3, j:j+3] += \
                        self.filters[f][np.newaxis, :, :, :] * \
                        dout[:, f, i, j][:, np.newaxis, np.newaxis, np.newaxis]
                    
        # Update weights
        self.filters -= self.learning_rate * dfilters / batch_size
        self.biases -= self.learning_rate * dbiases / batch_size
        
        return dX[:, :, 1:-1, 1:-1]


class PoolLayer:
    """Max pooling layer"""
    def forward(self, X):
        self.X = X
        batch_size, depth, height, width = X.shape
        output = np.zeros((batch_size, depth, height // 2, width // 2))
        
        for i in range(height // 2):
            for j in range(width // 2):
                X_slice = X[:, :, 2*i:2*i+2, 2*j:2*j+2]
                output[:, :, i, j] = np.max(X_slice, axis=(2, 3))
        
        return output
    
    def backward(self, dout):
        batch_size, depth, height, width = self.X.shape
        dX = np.zeros_like(self.X)
        
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                X_slice = self.X[:, :, 2*i:2*i+2, 2*j:2*j+2]
                max_indices = np.argmax(X_slice.reshape(batch_size, depth, -1), axis=2)
                for b in range(batch_size):
                    for d in range(depth):
                        idx = max_indices[b, d]
                        dX[b, d, 2*i + idx // 2, 2*j + idx % 2] = dout[b, d, i, j]
        
        return dX


class DenseLayer:
    """Fully connected layer"""
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.W = np.random.randn(input_size, output_size) * 0.01
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
        
        # Architecture
        self.conv1 = ConvLayer(num_filters, 1, learning_rate)
        self.pool1 = PoolLayer()
        self.conv2 = ConvLayer(num_filters * 2, num_filters, learning_rate)
        self.pool2 = PoolLayer()
        
        # Calculate flattened size
        self.flattened_size = num_filters * 2 * 7 * 7  # Assuming 28x28 input
        
        self.fc1 = DenseLayer(self.flattened_size, hidden_layers, learning_rate)
        self.fc2 = DenseLayer(hidden_layers, 10, learning_rate)
        
        # Store intermediate values for backward pass
        self.conv1_output = None
        self.pool1_output = None
        self.conv2_output = None
        self.pool2_output = None
        self.fc1_input = None
        self.fc1_output = None
        self.batch_size = None
        self.num_filters = num_filters
        
    def forward(self, X):
        self.batch_size = X.shape[0]
        X = self.conv1.forward(X)
        self.conv1_output = X.copy()
        X = np.maximum(X, 0)  # ReLU
        self.relu1_output = X.copy()
        
        X = self.pool1.forward(X)
        self.pool1_output = X.copy()
        
        X = self.conv2.forward(X)
        self.conv2_output = X.copy()
        X = np.maximum(X, 0)  # ReLU
        self.relu2_output = X.copy()
        
        X = self.pool2.forward(X)
        self.pool2_output = X.copy()
        
        X = X.reshape(X.shape[0], -1)
        self.fc1_input = X.copy()
        X = self.fc1.forward(X)
        self.fc1_output = X.copy()
        X = np.maximum(X, 0)  # ReLU
        self.relu3_output = X.copy()
        
        X = self.fc2.forward(X)
        return X
    
    def backward(self, dout):
        # FC2 backward
        dout = self.fc2.backward(dout)
        
        # ReLU backward
        dout[self.relu3_output <= 0] = 0
        
        # FC1 backward
        dout = self.fc1.backward(dout)
        
        # Reshape back to (batch_size, num_filters*2, 7, 7)
        dout = dout.reshape(self.batch_size, self.num_filters * 2, 7, 7)
        
        # Pool2 backward
        dout = self.pool2.backward(dout)
        
        # ReLU backward
        dout[self.relu2_output <= 0] = 0
        
        # Conv2 backward
        dout = self.conv2.backward(dout)
        
        # Pool1 backward
        dout = self.pool1.backward(dout)
        
        # ReLU backward
        dout[self.relu1_output <= 0] = 0
        
        # Conv1 backward
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
    # Load images
    with open(data_file, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_images = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_rows = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_cols = np.frombuffer(f.read(4), dtype='>i4')[0]
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    
    # Load labels
    with open(label_file, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype='>i4')[0]
        num_labels = np.frombuffer(f.read(4), dtype='>i4')[0]
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return images, labels


def train_cnn(dataset_path, labels_path, epochs, hidden_layers, callback=None):
    """
    Train a CNN model
    
    Args:
        dataset_path: Path to training images
        labels_path: Path to training labels
        epochs: Number of training epochs
        hidden_layers: Size of hidden layer
        callback: Function to call with progress updates
    
    Returns:
        Trained model
    """
    print(f"Loading data from {dataset_path} and {labels_path}...")
    
    # Load data
    try:
        images, labels = load_ubyte_data(dataset_path, labels_path)
    except Exception as e:
        print(f"Error loading ubyte files: {e}")
        raise
    
    # Normalize images
    images = images / 255.0
    
    # Reshape to (N, C, H, W) format
    images = images[:, np.newaxis, :, :]
    
    batch_size = 32
    num_samples = images.shape[0]
    
    # Initialize model
    model = SimpleCNN(num_filters=16, hidden_layers=hidden_layers, learning_rate=0.001)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(num_samples)
        images_shuffled = images[indices]
        labels_shuffled = labels[indices]
        
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            batch_images = images_shuffled[i:i+batch_size]
            batch_labels = labels_shuffled[i:i+batch_size]
            
            try:
                # Forward pass
                logits = model.forward(batch_images)
                loss, probs = cross_entropy_loss(logits, batch_labels)
                total_loss += loss
                num_batches += 1
                
                # Backward pass
                dout = probs.copy()
                dout[np.arange(batch_images.shape[0]), batch_labels] -= 1
                dout /= batch_images.shape[0]
                
                model.backward(dout)
            except Exception as e:
                print(f"Error in batch training: {e}")
                print(f"Batch shape: {batch_images.shape}")
                raise
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if callback:
            callback(epoch + 1, epochs, avg_loss)
    
    return model


def save_model(model, filepath):
    """Save model to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load model from file"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Example usage
    dataset_path = "/Users/noahj/Documents/KeineIntelligenz/dataset/mnist-dataset/train-images.idx3-ubyte"
    labels_path = "/Users/noahj/Documents/KeineIntelligenz/dataset/mnist-dataset/train-labels.idx1-ubyte"
    
    model = train_cnn(dataset_path, labels_path, epochs=2, hidden_layers=128)
    save_model(model, "cnn_model.pkl")

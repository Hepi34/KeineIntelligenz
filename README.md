# KeineIntelligenz

## Introduction
This is KeineIntelligenz, a program designed to train a CNN on the MNIST dataset. It is built in Python. Both CPU training and GPU training via OpenCL (PyOpenCL) are supported. The program was tested on Windows and macOS, and should work on Linux as well.

## Usage
To use the program, first clone this repository, then run `start.py` from the project root:

- Windows: `python start.py`
- macOS/Linux: `python3 start.py`

If needed, it will automatically create a virtual environment and install the required dependencies. After that, the main GUI will open.

## GUI

The GUI provides a user-friendly interface to train and test the neural network. You can create models, adjust training settings, and watch the training progress in real-time.

## Code documentation

### Files Overview

- **`start.py`** — Launches the program
- **`pyfiles/gui.py`** — The graphical user interface
- **`pyfiles/model.py`** — The neural network model
- **`pyfiles/trainer.py`** — Training logic
- **`pyfiles/dataset.py`** — Loads MNIST data
- **`pyfiles/layers.py`** — Building blocks (Conv, Dense, etc.)
- **`pyfiles/loss.py`** — Loss function for training
- **`pyfiles/optimizers.py`** — Optimizer algorithms (SGD, Adam)
- **`pyfiles/utils.py`** — Helper functions
- **`pyfiles/gpu_pipeline.py`** — GPU acceleration
- **`pyfiles/opencl_backend.py`** — OpenCL setup
- **`pyfiles/opencl_layers.py`** — GPU layer implementations

### File Descriptions

#### start.py

The main entry point that sets up everything before starting. It:
- Checks if a virtual environment exists (`.venv` folder)
- Creates one if it doesn't exist (using `python3 -m venv` on Mac/Linux or `python -m venv` on Windows)
- Reads `requirements.txt` and checks which packages are already installed
- Installs missing packages using pip
- Finally launches `pyfiles/gui.py` using the virtual environment's Python

Run this file whenever you want to start the GUI. It will set up missing dependencies automatically.

#### pyfiles/gui.py

The graphical interface built with PyQt6. It has:
- **Presets** - Pre-configured model setups (v1/Mini, v1/Normal, v1/Pro, v2/Mini, etc.) that you can quickly select. Each preset has different training settings, network size, and data limits.
- **Data Loading** - Automatically finds and loads MNIST images and labels from the dataset folder
- **Progress Display** - Shows training progress with graphs, loss curves, and accuracy over time
- **GPU Detection** - Automatically detects if you have a GPU available and can use it for faster training
- **Model Saving/Loading** - Save trained models as .npz files and load them back later
- Uses threading to keep the UI responsive while training happens in the background

#### pyfiles/model.py

The container that holds all your layers together to make a complete network. The `CNNModel` class does:
- **Forward Pass** - Takes input data and passes it through each layer in order (Conv2D → ReLU → MaxPool → ... → Dense)
- **Backward Pass** - Reverses through the layers to calculate gradients (used for updating weights)
- **Parameter Management** - Collects all weights and biases from all layers into one place so the optimizer can update them
- **Gradient Management** - Collects all gradients from all layers so the optimizer knows how much to change each weight

Example: if you have [Conv2D → ReLU → Dense], the model chains them together and handles the data flow.

#### pyfiles/trainer.py

Orchestrates the entire training process. It has:
- **TrainConfig** - Stores settings like number of epochs, batch size, whether to shuffle data, and thread count
- **Trainer Class** - Does the actual training with these steps:
  1. Loop through each epoch
  2. For each batch of data: do forward pass, calculate loss, do backward pass, update weights
  3. After each epoch: evaluate on test data and record accuracy
  4. Return history of losses and accuracies
- **Batch Iterator** - Splits data into small chunks (batches) for training
- **Accuracy Calculation** - Checks how many predictions are correct
- **Thread Management** - Can use multiple CPU cores for faster numpy operations

#### pyfiles/dataset.py

Handles loading the MNIST handwritten digit dataset. It:
- **Reads IDX Format** - MNIST files are in a special binary format (.ubyte files). This code reads:
  - Magic numbers to verify file type
  - Image count, dimensions (28×28 for MNIST)
  - Raw pixel bytes (one byte per pixel = 0-255)
  - Label bytes (0-9 for digits)
- **Normalizes Data** - Converts pixel values from 0-255 to 0-1 by dividing by 255 (helps training)
- **Reshapes Images** - Converts flat byte arrays into 4D arrays (N, 1, 28, 28) where N = number of images
- **Splits Data** - Returns train images, train labels, test images, test labels as separate arrays
- **MNISTDataset Class** - Simple container to bundle all four arrays together

#### pyfiles/layers.py

Defines the building blocks of the neural network. Each layer has `forward()` (compute output) and `backward()` (compute gradients) methods:

- **Conv2D** - Convolutional layer that finds patterns in images:
  - Uses 3×3 filters that slide over the image
  - Has stride (how far the filter moves) and padding options
  - Stores weights and biases that get updated during training
  - Uses "He initialization" to properly scale starting weights

- **Dense** - Fully connected layer that connects every input to every output:
  - Like a big matrix multiplication with bias added
  - Usually used at the end before final predictions
  - Takes 2D input (batch, features) and outputs (batch, output_size)

- **ReLU** - Activation function that adds non-linearity:
  - Simple: outputs max(0, input) 
  - Turns negative values to 0, keeps positive values
  - Makes the network able to learn non-linear patterns

- **MaxPool2D** - Reduces size while keeping important info:
  - Slides a window (like 2×2) over the image
  - Keeps only the maximum value in each window
  - Reduces computation and helps prevent overfitting

- **Flatten** - Converts 2D/3D data to 1D:
  - Takes (N, C, H, W) and reshapes to (N, C*H*W)
  - Bridge between convolutional layers and fully connected layers

- **Dropout** - Randomly drops neurons during training:
  - Helps prevent overfitting by "thinning" the network
  - Only active during training, disabled during testing
  - Scales remaining activations to compensate

#### pyfiles/loss.py

Measures how wrong your network's predictions are. It has:

- **CrossEntropy** - The loss function used for digit classification:
  - Takes raw predictions (logits) from the network and true digit labels (0-9)
  - Computes softmax (converts to probabilities that sum to 1)
  - Calculates how different predicted probabilities are from the truth (high loss = bad predictions)
  - **Numerical Stability** - Uses a trick (subtracting max before exp) to prevent overflow errors in math
  - **Backward** - Computes gradients needed to update network weights
  - Lower loss = better predictions = happy training!

#### pyfiles/optimizers.py

Contains algorithms that improve the network during training by updating weights:

- **SGD (Stochastic Gradient Descent)** - Simple optimizer:
  - Updates weights by: `weight -= learning_rate * gradient`
  - All parameters get the same treatment
  - Can add weight decay (L2 regularization) to prevent overfitting
  - Fast but sometimes needs careful tuning

- **Adam** - More advanced optimizer:
  - Adapts learning rate for each parameter individually
  - Keeps momentum (remembers previous updates)
  - Usually needs less tuning than SGD
  - Better convergence in practice
  - Has parameters: beta1 (momentum for gradients), beta2 (momentum for squared gradients), epsilon (numerical stability)

Both can be used by passing them to the Trainer along with a learning rate.

#### pyfiles/utils.py

Helper functions and utilities used throughout training:

- **accuracy_score()** - Computes classification accuracy from predicted labels and true labels.

- **batch_iterator()** - Creates mini-batches from training data, optionally shuffling before batching.

- **Timer** - Simple utility to measure how long things take:
  - `start()` - begin timing
  - `stop()` - end timing and get elapsed seconds
  - Useful for measuring epoch time or total training time

#### pyfiles/gpu_pipeline.py

A complete GPU-based training pipeline using OpenCL. It includes:

- **OpenCL Kernels** - Low-level GPU code written in C-like language that runs on the GPU:
  - `conv2d_forward` - Convolution forward pass optimized for GPU
  - `conv2d_backward` - Convolution backward pass for gradients
  - `dense_forward` - Fully connected layer forward
  - `dense_backward` - Fully connected layer backward
  - `copy_batch` - Fast memory copy for batches
  - Other kernels for activation functions, pooling, dropout

- **GPUTrainConfig** - Settings for GPU training (epochs, batch size, layer sizes, optimizer choice, etc.)

- **GPUTrainingPipeline** - The main GPU trainer that:
  - Compiles OpenCL kernels once at startup
  - Transfers data between CPU and GPU memory
  - Launches GPU kernels for forward/backward passes
  - Handles parameter updates on GPU
  - Much faster than CPU for large models (but overhead for small ones)

This file is an all-GPU implementation, whereas the CPU version uses the regular trainer.py.

#### pyfiles/opencl_backend.py

Sets up OpenCL for GPU acceleration. It:

- **Imports PyOpenCL** - The library that talks to GPUs
- **GPUDeviceInfo** - Data class storing GPU information (platform name, device name, vendor, driver version)
- **OpenCLManager** - Main class that:
  - Detects available GPU devices (NVIDIA, AMD, Intel, Apple Silicon, etc.)
  - Creates an OpenCL context (connection to the GPU)
  - Creates a command queue (sends commands to GPU)
  - Provides utility methods for transferring tensors to/from GPU memory
  - Owns buffers for storing data on the GPU

When the program starts, it tries to detect a GPU. If one is found, the GUI can offer GPU training as an option.

#### pyfiles/opencl_layers.py

A small helper module for OpenCL-backed layer implementations.

- Provides an `OpenCLConv2D` layer that runs a Conv2D forward pass on the GPU via a compiled OpenCL kernel
- Includes a minimal OpenCL kernel for Conv2D; it does not implement full backward passes or other layer types (Dense/ReLU/MaxPool) in this module
- Intended as a building block for GPU-accelerated operations, but the main GPU training pipeline is in `pyfiles/gpu_pipeline.py`

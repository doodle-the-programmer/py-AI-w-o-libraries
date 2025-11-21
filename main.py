"""
Neural Network from Scratch - Vectorized Implementation

This is a complete neural network implementation built from first principles using
only NumPy for numerical operations. It demonstrates fundamental deep learning
concepts including:
- Feedforward propagation
- Backpropagation with gradient descent
- Adam optimizer (adaptive learning rates)
- Data augmentation for better generalization
- Ensemble learning (multiple networks voting)
- Early stopping to prevent overfitting

No high-level ML frameworks like TensorFlow, PyTorch, Keras, or Pandas are used.
This provides full transparency into how neural networks actually work.
"""

import random
import math
import multiprocessing as mp
from typing import List, Tuple, Callable, Optional
from collections import defaultdict
import time

# Import NumPy for vectorized operations
# Vectorization means operating on entire arrays at once rather than element-by-element
# This leverages optimized C/Fortran code under the hood for 10-100x speedup
try:
    import numpy as np
    print("⚡ NumPy acceleration enabled")
except ImportError:
    print("❌ NumPy not found. Please install it:")
    print("   pip install numpy")
    exit(1)

# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================
# Activation functions introduce non-linearity into the network, allowing it
# to learn complex patterns. Without them, the network would just be a series
# of matrix multiplications (linear transformations), which could only learn
# linear relationships.
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation for multi-class classification.
    
    Converts raw output scores (logits) into probabilities that sum to 1.
    Each output represents the probability of that class being correct.
    
    Mathematical formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    Numerical stability trick: We subtract the max value before exponentiating
    to prevent overflow (exp(large_number) can exceed float limits).
    This doesn't change the result since: exp(x-c) / sum(exp(x-c)) = exp(x) / sum(exp(x))
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)  # Prevent overflow
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
    
    Squashes input values to range (0, 1), making them interpretable as probabilities.
    Used in the output layer before softmax to normalize the raw neuron outputs.
    
    Properties:
    - Output range: (0, 1)
    - Smooth, differentiable curve
    - Centers around 0.5
    
    We clip input to [-500, 500] to prevent overflow in exp(-x).
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function: σ'(x) = σ(x) * (1 - σ(x))
    
    Used during backpropagation to compute gradients.
    This efficient form avoids recomputing the exponential.
    """
    s = sigmoid(x)
    return s * (1 - s)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU (Rectified Linear Unit) activation function.
    
    Formula: f(x) = x if x > 0, else alpha * x
    
    This is an improved version of ReLU that addresses the "dying ReLU" problem
    where neurons can get stuck outputting 0 during training. By allowing a small
    negative slope (alpha), gradients can still flow backwards even for negative inputs.
    
    Benefits over standard ReLU:
    - Prevents "dead neurons" that never activate
    - Allows negative information to pass through (attenuated)
    - Faster training than sigmoid/tanh (simpler computation)
    
    Typical alpha values: 0.01 to 0.3
    """
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of Leaky ReLU: f'(x) = 1 if x > 0, else alpha
    
    Used in backpropagation to compute gradients through hidden layers.
    The derivative is constant in each region (piecewise linear function).
    """
    return np.where(x > 0, 1.0, alpha)


class NeuralNetworkVectorized:
    """
    Fully vectorized neural network using NumPy for high performance.
    
    Architecture:
    - Feedforward network with arbitrary hidden layers
    - Uses Leaky ReLU activation in hidden layers
    - Uses Sigmoid + Softmax in output layer for multi-class classification
    - Implements Adam optimizer for adaptive learning rates
    
    The network processes data in batches using matrix operations, which is
    much faster than processing one sample at a time due to NumPy's optimized
    linear algebra routines (BLAS/LAPACK).
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.05,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize vectorized neural network with He initialization and Adam optimizer.
        
        Args:
            layer_sizes: List of neurons in each layer, e.g., [9, 24, 12, 3]
                        First value is input size, last is output size (num classes)
            learning_rate: Step size for weight updates (higher = faster but less stable)
            beta1: Adam exponential decay rate for first moment estimates (momentum)
                   Typical value: 0.9 (keeps 90% of previous gradient direction)
            beta2: Adam exponential decay rate for second moment estimates (variance)
                   Typical value: 0.999 (tracks gradient variance for adaptive rates)
            epsilon: Small constant to prevent division by zero in Adam (1e-8 is standard)
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Adam time step (increments each batch, used for bias correction)
        
        # Initialize weights and biases for each layer
        self.weights = []  # List of weight matrices
        self.biases = []   # List of bias vectors
        
        # Adam optimizer state variables
        # Adam maintains running averages of gradients (momentum) and squared gradients (variance)
        self.m_weights = []  # First moment (mean) for weights
        self.v_weights = []  # Second moment (variance) for weights
        self.m_biases = []   # First moment for biases
        self.v_biases = []   # Second moment for biases
        
        for i in range(len(layer_sizes) - 1):
            # He initialization: optimal for ReLU-like activations
            # Formula: weights ~ U(-sqrt(2/n_in), +sqrt(2/n_in))
            # This prevents vanishing/exploding gradients by scaling based on layer size
            limit = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            
            # Biases typically start at zero (weights handle the initial randomness)
            b = np.zeros(layer_sizes[i + 1])
            
            self.weights.append(W)
            self.biases.append(b)
            
            # Initialize Adam state variables to zero
            # These will accumulate gradient information during training
            self.m_weights.append(np.zeros_like(W))
            self.v_weights.append(np.zeros_like(W))
            self.m_biases.append(np.zeros_like(b))
            self.v_biases.append(np.zeros_like(b))
        
        # Cache for backpropagation
        # We store intermediate values during forward pass to use in backward pass
        self.activations = []    # Output of each layer after activation function
        self.weighted_sums = []  # Pre-activation values (z = Wx + b)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation: compute network output from input.
        
        Process:
        1. For each hidden layer: compute z = Wx + b, then apply Leaky ReLU
        2. For output layer: compute z = Wx + b, apply sigmoid, then softmax
        3. Store intermediate values (needed for backpropagation)
        
        Mathematical notation:
        - a^[l] = activation output of layer l
        - z^[l] = weighted sum (pre-activation) of layer l
        - W^[l] = weight matrix connecting layer l-1 to layer l
        - b^[l] = bias vector for layer l
        
        Formula: z^[l] = W^[l] * a^[l-1] + b^[l]
                 a^[l] = activation_function(z^[l])
        
        Args:
            X: Input array of shape (batch_size, input_dim) or (input_dim,)
               Can be a single sample or batch of samples
        
        Returns:
            Output probabilities of shape (batch_size, output_dim) or (output_dim,)
            Values sum to 1.0 across the output dimension (thanks to softmax)
        """
        # Handle single sample by adding batch dimension
        single_sample = X.ndim == 1
        if single_sample:
            X = X.reshape(1, -1)  # Convert (9,) to (1, 9)
        
        # Initialize storage for forward pass values
        self.activations = [X]  # Store input as first activation
        self.weighted_sums = []
        
        # Process through hidden layers
        # Each layer: linear transformation (Wx+b) followed by non-linear activation
        for i in range(len(self.weights) - 1):
            # Matrix multiplication: (batch_size, n_in) @ (n_in, n_out) = (batch_size, n_out)
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.weighted_sums.append(z)
            
            # Apply Leaky ReLU activation (introduces non-linearity)
            a = leaky_relu(z)
            self.activations.append(a)
        
        # Output layer: use sigmoid for normalized values, then softmax for probabilities
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.weighted_sums.append(z)
        a = sigmoid(z)  # Normalize to (0, 1) range
        self.activations.append(a)
        
        # Apply softmax to convert to probability distribution
        # Output will sum to 1.0, representing class probabilities
        output = softmax(a)
        
        # Return to original shape if input was single sample
        if single_sample:
            return output.flatten()
        return output
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Backpropagation: compute gradients and update weights using Adam optimizer.
        
        This is where the network "learns" by:
        1. Computing the error (how wrong the predictions are)
        2. Calculating gradients (how to change weights to reduce error)
        3. Updating weights using Adam optimizer (adaptive learning rates)
        
        Backpropagation works by applying the chain rule of calculus backwards
        through the network, computing how each weight contributed to the error.
        
        Adam optimizer improvements over standard gradient descent:
        - Adaptive learning rates per parameter
        - Momentum to accelerate convergence
        - Bias correction for early training steps
        
        Args:
            X: Input array of shape (batch_size, input_dim)
            y: Target one-hot encoded labels of shape (batch_size, output_dim)
               e.g., [1, 0, 0] for class 0, [0, 1, 0] for class 1, etc.
        
        Returns:
            Cross-entropy loss value (lower is better)
        """
        # Forward pass to get predictions and cache intermediate values
        output = self.forward(X)
        
        # Ensure inputs are 2D for consistent batch processing
        if y.ndim == 1:
            y = y.reshape(1, -1)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        batch_size = X.shape[0]
        
        # Cross-entropy loss: measures how different predictions are from true labels
        # Formula: L = -sum(y * log(ŷ)) / batch_size
        # We clip output to prevent log(0) which would give -infinity
        # Lower loss = better predictions
        loss = -np.sum(y * np.log(np.clip(output, 1e-10, 1.0))) / batch_size
        
        # Increment Adam time step for bias correction
        # t=1 on first update, t=2 on second, etc.
        self.t += 1
        
        # === BACKPROPAGATION: Compute gradients layer by layer ===
        
        # Output layer gradient
        # For softmax + cross-entropy, the derivative simplifies to: δ = ŷ - y
        # This elegant result comes from the mathematical properties of these functions
        delta = output - y  # Shape: (batch_size, output_dim)
        
        # Propagate error backwards through hidden layers
        # Each delta represents how much each neuron contributed to the final error
        deltas = [delta]
        for i in range(len(self.weights) - 1, 0, -1):
            # Chain rule: δ^[l-1] = (δ^[l] * W^[l]^T) ⊙ f'(z^[l-1])
            # - δ^[l] * W^[l]^T: propagate error back through weights
            # - ⊙ f'(z^[l-1]): multiply by activation derivative (element-wise)
            delta = (delta @ self.weights[i].T) * leaky_relu_derivative(self.weighted_sums[i - 1])
            deltas.insert(0, delta)  # Prepend so deltas[i] corresponds to layer i
        
        # === ADAM OPTIMIZER: Update weights and biases ===
        # Adam = Adaptive Moment Estimation
        # Combines momentum (first moment) and RMSprop (second moment)
        
        for i in range(len(self.weights)):
            # Compute gradients
            # Weight gradient: ∂L/∂W = a^[l-1]^T * δ^[l] / batch_size
            # This tells us how to change each weight to reduce loss
            grad_W = self.activations[i].T @ deltas[i] / batch_size
            
            # Bias gradient: ∂L/∂b = sum(δ^[l]) / batch_size
            # Average error signal for each neuron
            grad_b = np.sum(deltas[i], axis=0) / batch_size
            
            # === Adam update for weights ===
            # First moment (momentum): exponential moving average of gradients
            # Helps accelerate in consistent gradient directions
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * grad_W
            
            # Second moment (variance): exponential moving average of squared gradients
            # Used to normalize updates (larger for parameters with smaller gradients)
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * grad_W ** 2
            
            # Bias correction: counteract initialization bias (m and v start at 0)
            # Without this, early updates would be biased toward zero
            m_hat_W = self.m_weights[i] / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_weights[i] / (1 - self.beta2 ** self.t)
            
            # Final update: learning_rate * momentum / sqrt(variance)
            # Each parameter gets its own adaptive learning rate
            self.weights[i] -= self.learning_rate * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
            
            # === Adam update for biases (same process) ===
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * grad_b
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * grad_b ** 2
            
            m_hat_b = self.m_biases[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        
        return float(loss)
    
    def get_weights(self):
        """Get copy of all weights for serialization or ensemble averaging."""
        return [w.copy() for w in self.weights]
    
    def set_weights(self, weights):
        """Set weights from external source (for loading saved models)."""
        self.weights = [np.array(w) for w in weights]


# ============================================================================
# DATA AUGMENTATION
# ============================================================================
# Data augmentation artificially increases the training set size by creating
# modified versions of existing samples. This helps the network generalize
# better and reduces overfitting, especially important for small datasets.
#
# For our 3x3 grid patterns, we use geometric transformations that preserve
# the shape identity (an "L" rotated is still an "L").
def rotate_grid_90(grid: np.ndarray) -> np.ndarray:
    """Rotate 3x3 grid 90 degrees clockwise."""
    return np.array([grid[6], grid[3], grid[0],
                     grid[7], grid[4], grid[1],
                     grid[8], grid[5], grid[2]])

def rotate_grid_180(grid: np.ndarray) -> np.ndarray:
    """Rotate 3x3 grid 180 degrees."""
    return grid[::-1]

def rotate_grid_270(grid: np.ndarray) -> np.ndarray:
    """Rotate 3x3 grid 270 degrees clockwise."""
    return np.array([grid[2], grid[5], grid[8],
                     grid[1], grid[4], grid[7],
                     grid[0], grid[3], grid[6]])

def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    """Flip 3x3 grid horizontally."""
    return np.array([grid[2], grid[1], grid[0],
                     grid[5], grid[4], grid[3],
                     grid[8], grid[7], grid[6]])

def flip_vertical(grid: np.ndarray) -> np.ndarray:
    """Flip 3x3 grid vertically."""
    return np.array([grid[6], grid[7], grid[8],
                     grid[3], grid[4], grid[5],
                     grid[0], grid[1], grid[2]])

def augment_data(data: List[Tuple[List[int], List[int]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to increase training set size.
    
    For each sample, we create 6 variations:
    1. Original
    2-4. Rotations (90°, 180°, 270°)
    5-6. Flips (horizontal, vertical)
    
    This increases dataset size by 6x (from ~18 to ~108 samples), helping the
    network learn rotation-invariant and flip-invariant features.
    
    Returns:
        Tuple of (augmented_inputs, augmented_targets) as numpy arrays
        Both have shape (num_samples * 6, feature_dim)
    """
    augmented_inputs = []
    augmented_targets = []
    
    for inputs, target in data:
        inputs_arr = np.array(inputs)
        target_arr = np.array(target)
        
        # Original sample
        augmented_inputs.append(inputs_arr)
        augmented_targets.append(target_arr)
        
        # Rotations: teach network that shape orientation doesn't matter
        augmented_inputs.append(rotate_grid_90(inputs_arr))
        augmented_targets.append(target_arr)
        
        augmented_inputs.append(rotate_grid_180(inputs_arr))
        augmented_targets.append(target_arr)
        
        augmented_inputs.append(rotate_grid_270(inputs_arr))
        augmented_targets.append(target_arr)
        
        # Flips: additional geometric variations
        augmented_inputs.append(flip_horizontal(inputs_arr))
        augmented_targets.append(target_arr)
        
        augmented_inputs.append(flip_vertical(inputs_arr))
        augmented_targets.append(target_arr)
    
    return np.array(augmented_inputs), np.array(augmented_targets)


# Training data
base_training_data = [
    # L shapes
    [[1, 0, 0, 1, 0, 0, 1, 1, 1], [1, 0, 0]],
    [[1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 0, 0]],
    [[0, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 0]],
    [[1, 1, 0, 1, 0, 0, 1, 0, 0], [1, 0, 0]],
    [[1, 1, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0]],
    
    # O shapes
    [[1, 1, 1, 1, 0, 1, 1, 1, 1], [0, 1, 0]],
    [[1, 1, 0, 1, 1, 0, 0, 0, 0], [0, 1, 0]],
    [[0, 1, 1, 0, 1, 1, 0, 0, 0], [0, 1, 0]],
    [[0, 0, 0, 1, 1, 0, 1, 1, 0], [0, 1, 0]],
    [[0, 0, 0, 0, 1, 1, 0, 1, 1], [0, 1, 0]],
    [[1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0]],
    [[0, 1, 0, 1, 1, 1, 0, 1, 0], [0, 1, 0]],
    
    # > shapes
    [[1, 1, 0, 0, 1, 1, 1, 1, 0], [0, 0, 1]],
    [[1, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 1]],
    [[0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1]],
    [[1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 1]],
    [[1, 0, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1]],
    [[1, 0, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1]],
]


class AcceleratedEnsemble:
    """
    Ensemble of multiple neural networks for improved predictions.
    
    Ensemble learning combines predictions from multiple independent models,
    which typically outperforms any single model. Benefits:
    - Reduces overfitting (averaging smooths out individual model errors)
    - More robust predictions (less sensitive to random initialization)
    - Better generalization (different networks learn different features)
    
    The ensemble predicts by averaging the output probabilities of all networks,
    effectively creating a "committee vote" for each prediction.
    """
    
    def __init__(self, num_networks: int, layer_sizes: List[int], 
                 learning_rate: float = 0.1, use_multiprocessing: bool = True):
        self.num_networks = num_networks
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.use_multiprocessing = use_multiprocessing and mp.cpu_count() > 1
        
        # Create independent networks with different random initializations
        # Each network will learn slightly different features due to:
        # 1. Random weight initialization
        # 2. Random batch ordering during training
        # 3. Random data shuffling
        self.networks = [
            NeuralNetworkVectorized(layer_sizes, learning_rate=learning_rate)
            for _ in range(num_networks)
        ]
        
        # Track training metrics over time
        self.training_history = defaultdict(list)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction by averaging outputs from all networks in ensemble.
        
        This "wisdom of crowds" approach often beats individual predictions:
        - Random errors cancel out when averaged
        - Consistent patterns are reinforced
        - More confident when all networks agree
        """
        predictions = [net.forward(X) for net in self.networks]
        return np.mean(predictions, axis=0)  # Average across networks
    
    def train(self, training_data: List[Tuple[List[int], List[int]]], 
              epochs: int = 500, batch_size: int = 32, validation_split: float = 0.15):
        """
        Train all networks in the ensemble with mini-batch gradient descent.
        
        Training process:
        1. Data augmentation: expand dataset 6x with geometric transformations
        2. Train/validation split: reserve some data for unbiased evaluation
        3. Mini-batch training: process small batches for faster convergence
        4. Early stopping: halt training if validation accuracy stops improving
        
        Mini-batching benefits:
        - Faster updates (don't wait for entire dataset)
        - Regularization effect (noise in batches prevents overfitting)
        - Better memory efficiency (process subset of data at once)
        
        Args:
            training_data: List of (input, target) tuples (will be augmented)
            epochs: Maximum number of passes through the dataset
            batch_size: Number of samples per gradient update
            validation_split: Fraction of augmented data to reserve for validation (0.15 = 15%)
        """
        print(f"\n{'='*60}")
        print("NEURAL NETWORK TRAINING")
        print(f"{'='*60}")
        print(f"Backend: NumPy (CPU-optimized)")
        print(f"Networks: {self.num_networks}")
        print(f"Architecture: {self.layer_sizes}")
        print(f"Multiprocessing: {'Enabled' if self.use_multiprocessing else 'Disabled'}")
        print(f"{'='*60}\n")
        
        # Apply data augmentation (rotations, flips) to increase dataset size
        X_all, y_all = augment_data(training_data)
        
        # Shuffle data randomly to prevent order bias
        # Important: prevents network from learning dataset ordering patterns
        indices = np.random.permutation(len(X_all))
        X_all = X_all[indices]
        y_all = y_all[indices]
        
        # Split into training and validation sets
        # Training set: used to update weights
        # Validation set: used to monitor generalization (never seen during weight updates)
        split_idx = int(len(X_all) * (1 - validation_split))
        X_train, y_train = X_all[:split_idx], y_all[:split_idx]
        X_val, y_val = X_all[split_idx:], y_all[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {batch_size}")
        print(f"{'-'*60}\n")
        
        # Early stopping variables
        # Stop training if validation accuracy doesn't improve for max_patience epochs
        best_val_acc = 0      # Best validation accuracy seen so far
        patience = 0          # Number of epochs without improvement
        max_patience = 50     # Stop if no improvement for this many epochs
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Shuffle training data each epoch for better generalization
            # Different orderings expose the network to different batch combinations
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Mini-batch training: split data into small batches
            total_loss = 0
            num_batches = (len(X_train) + batch_size - 1) // batch_size  # Ceiling division
            
            # Process each batch
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]  # Get batch of inputs
                y_batch = y_train[i:i+batch_size]  # Get batch of targets
                
                # Train all networks in ensemble on this batch
                # Each network learns independently from the same data
                for network in self.networks:
                    loss = network.backward(X_batch, y_batch)  # Compute gradients and update
                    total_loss += loss
            
            # Average loss across all batches and networks
            avg_loss = total_loss / (num_batches * self.num_networks)
            
            # Evaluate performance every 10 epochs (saves computation time)
            if epoch % 10 == 0 or epoch == epochs - 1:
                # Calculate accuracy on both training and validation sets
                train_acc = self._calculate_accuracy(X_train, y_train)
                val_acc = self._calculate_accuracy(X_val, y_val)
                
                elapsed = time.time() - start_time
                
                # Store metrics for later analysis
                self.training_history['epoch'].append(epoch)
                self.training_history['loss'].append(avg_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_acc'].append(val_acc)
                
                # Display progress
                print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | "
                      f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                      f"Time: {elapsed:.1f}s")
                
                # Early stopping: prevent overfitting by stopping when validation stops improving
                # If validation accuracy improves, reset patience counter
                # If not improving, increment patience
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience = 0  # Reset patience when we improve
                else:
                    patience += 1  # Increment patience when no improvement
                
                # Stop training if we've waited too long without improvement
                if patience >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best validation accuracy: {best_val_acc:.1f}%")
                    break  # Exit training loop
        
        total_time = time.time() - start_time
        print(f"\n{'-'*60}")
        print(f"Training complete in {total_time:.1f}s")
        print(f"Final validation accuracy: {val_acc:.1f}%")
        print(f"{'='*60}\n")
    
    def _calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate classification accuracy as percentage of correct predictions.
        
        Process:
        1. Get ensemble predictions (probability distributions)
        2. Take argmax to get predicted class (highest probability)
        3. Compare with true class (argmax of one-hot encoding)
        4. Return percentage of matches
        
        Returns:
            Accuracy as a percentage (0-100)
        """
        predictions = self.forward(X)  # Get ensemble predictions
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
        
        # Convert probability distributions to class labels
        pred_classes = np.argmax(predictions, axis=1)  # Predicted class
        true_classes = np.argmax(y, axis=1)            # True class
        
        # Calculate percentage of correct predictions
        return float(np.mean(pred_classes == true_classes) * 100)


def visualize_grid(grid: np.ndarray) -> None:
    """Display a 3x3 grid."""
    symbols = {0: '⬜', 1: '⬛'}
    print("\n  Grid visualization:")
    for i in range(3):
        row = grid[i*3:i*3+3]
        print("  " + " ".join(symbols[int(cell)] for cell in row))
    print()


def interactive_mode(ensemble: AcceleratedEnsemble):
    """Interactive prediction mode."""
    shape_names = {0: "L", 1: "O", 2: ">"}
    
    print(f"\n{'='*60}")
    print("INTERACTIVE PREDICTION MODE")
    print(f"{'='*60}")
    print("Enter a 3x3 grid pattern to predict the shape.")
    print("Enter 1 for black (⬛) and 0 for white (⬜)")
    print("Type 'quit' to exit")
    print(f"{'-'*60}\n")
    
    try:
        while True:
            print("\nEnter grid values (top-left to bottom-right):")
            uinput = []
            
            try:
                for i in range(9):
                    while True:
                        value = input(f"  Square #{i+1}: ").strip().lower()
                        if value == 'quit':
                            print("\nExiting interactive mode. Goodbye!")
                            return
                        if value in ['0', '1']:
                            uinput.append(int(value))
                            break
                        print("    Invalid input. Enter 0 or 1.")
                
                # Get prediction
                grid = np.array(uinput)
                output = ensemble.forward(grid)
                predicted_class = int(np.argmax(output))
                prediction = shape_names[predicted_class]
                confidence = float(output[predicted_class]) * 100
                
                # Display results
                visualize_grid(grid)
                print(f"🎯 Predicted shape: {prediction}")
                print(f"📊 Confidence: {confidence:.2f}%")
                print(f"📈 Class probabilities:")
                for i, prob in enumerate(output):
                    print(f"   {shape_names[i]}: {float(prob)*100:.2f}%")
                print(f"{'-'*60}")
                
            except ValueError as e:
                print(f"Error: {e}")
                continue
                
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting...")
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Network configuration
    layer_sizes = [9, 24, 12, 3]
    
    # Create and train ensemble
    print(f"\n{'='*60}")
    print("NEURAL NETWORK - NUMPY VECTORIZED VERSION")
    print(f"{'='*60}")
    
    ensemble = AcceleratedEnsemble(
        num_networks=3,
        layer_sizes=layer_sizes,
        learning_rate=0.1,
        use_multiprocessing=False
    )
    
    # Train
    ensemble.train(base_training_data, epochs=500, batch_size=16)
    
    # Interactive mode
    interactive_mode(ensemble)

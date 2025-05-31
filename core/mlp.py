import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable
from core.gradient_descent import GradientDescent

# Under testing
class MLP:
    """
    Multi-Layer Perceptron that's compatible with the custom gradient descent algorithm.
    
    This MLP flattens all weights and biases into a single parameter vector to work
    with the existing GradientDescent class that expects a 1D weight array.
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu'):
        """
        Initialize the MLP.
        
        Args:
            layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.num_layers = len(layer_sizes) - 1
        
        # Store layer dimensions for weight reconstruction
        self.weight_shapes = []
        self.bias_shapes = []
        self.weight_indices = []  # Start and end indices for each layer's weights
        self.bias_indices = []    # Start and end indices for each layer's biases
        
        # Calculate shapes and indices
        self._calculate_parameter_layout()
        
        # Total number of parameters
        self.total_params = self._calculate_total_params()
        
        # Initialize weights
        self.weights = self._initialize_weights()
        
    def _calculate_parameter_layout(self):
        """Calculate the shapes and indices for weights and biases."""
        current_idx = 0
        
        for i in range(self.num_layers):
            input_dim = self.layer_sizes[i]
            output_dim = self.layer_sizes[i + 1]
            
            # Weight shape: (input_dim, output_dim)
            weight_shape = (input_dim, output_dim)
            self.weight_shapes.append(weight_shape)
            
            # Bias shape: (output_dim,)
            bias_shape = (output_dim,)
            self.bias_shapes.append(bias_shape)
            
            # Weight indices
            weight_size = input_dim * output_dim
            self.weight_indices.append((current_idx, current_idx + weight_size))
            current_idx += weight_size
            
            # Bias indices
            bias_size = output_dim
            self.bias_indices.append((current_idx, current_idx + bias_size))
            current_idx += bias_size
    
    def _calculate_total_params(self):
        """Calculate total number of parameters."""
        total = 0
        for i in range(self.num_layers):
            # Weights
            total += self.layer_sizes[i] * self.layer_sizes[i + 1]
            # Biases
            total += self.layer_sizes[i + 1]
        return total
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        weights = np.zeros(self.total_params)
        
        for i in range(self.num_layers):
            input_dim = self.layer_sizes[i]
            output_dim = self.layer_sizes[i + 1]
            
            # Xavier initialization for weights
            weight_start, weight_end = self.weight_indices[i]
            weight_size = weight_end - weight_start
            
            # Standard Xavier initialization: sqrt(6 / (fan_in + fan_out))
            limit = np.sqrt(6.0 / (input_dim + output_dim))
            weights[weight_start:weight_end] = np.random.uniform(
                -limit, limit, size=weight_size
            )
            
            # Initialize biases to zero
            bias_start, bias_end = self.bias_indices[i]
            weights[bias_start:bias_end] = 0.0
            
        return weights
    
    def _weights_to_layers(self, weights: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Convert flattened weights back to layer weights and biases."""
        layer_weights = []
        layer_biases = []
        
        for i in range(self.num_layers):
            # Extract weights
            weight_start, weight_end = self.weight_indices[i]
            weight_flat = weights[weight_start:weight_end]
            weight_matrix = weight_flat.reshape(self.weight_shapes[i])
            layer_weights.append(weight_matrix)
            
            # Extract biases
            bias_start, bias_end = self.bias_indices[i]
            bias_vector = weights[bias_start:bias_end]
            layer_biases.append(bias_vector)
            
        return layer_weights, layer_biases
    
    def _get_activation_function(self):
        """Get the activation function."""
        if self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def forward(self, x: np.ndarray, weights: np.ndarray) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input data (n_samples, input_dim)
            weights: Flattened weight vector
            
        Returns:
            Output predictions as torch tensor
        """
        # Convert input to tensor
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x
            
        # Convert weights to layer format
        layer_weights, layer_biases = self._weights_to_layers(weights)
        
        # Forward pass
        activation_fn = self._get_activation_function()
        current_input = x_tensor
        
        for i in range(self.num_layers):
            # Convert numpy arrays to tensors
            weight_tensor = torch.tensor(layer_weights[i], dtype=torch.float32)
            bias_tensor = torch.tensor(layer_biases[i], dtype=torch.float32)
            
            # Linear transformation: y = xW + b
            linear_output = torch.matmul(current_input, weight_tensor) + bias_tensor
            
            # Apply activation (except for the last layer)
            if i < self.num_layers - 1:
                current_input = activation_fn(linear_output)
            else:
                current_input = linear_output  # No activation on output layer
                
        return current_input
    
    def predict(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions with the current or provided weights.
        
        Args:
            x: Input data (n_samples, input_dim)
            weights: Optional weight vector, uses self.weights if None
            
        Returns:
            Predictions as numpy array
        """
        if weights is None:
            weights = self.weights
            
        with torch.no_grad():
            output = self.forward(x, weights)
            return output.numpy()
    
    def create_loss_function(self, x: np.ndarray, y: np.ndarray, loss_type: str = 'mse') -> Callable:
        """
        Create a loss function compatible with the gradient descent algorithm.
        
        Args:
            x: Input features (n_samples, input_dim)
            y: Target values (n_samples, output_dim) or (n_samples,)
            loss_type: Type of loss ('mse', 'cross_entropy', 'binary_cross_entropy')
            
        Returns:
            Loss function that takes weights and returns scalar loss
        """
        def loss_function(weights: torch.Tensor) -> torch.Tensor:
            # Forward pass
            predictions = self.forward(x, weights.detach().numpy())
            
            # Convert targets to tensor
            y_tensor = torch.tensor(y, dtype=torch.float32)
            
            # Ensure predictions require gradients
            predictions = predictions.requires_grad_(True)
            
            # Compute loss based on type
            if loss_type == 'mse':
                if y_tensor.dim() == 1 and predictions.dim() == 2 and predictions.shape[1] == 1:
                    predictions = predictions.squeeze()
                loss = torch.mean((predictions - y_tensor) ** 2)
            elif loss_type == 'cross_entropy':
                loss = F.cross_entropy(predictions, y_tensor.long())
            elif loss_type == 'binary_cross_entropy':
                predictions = torch.sigmoid(predictions)
                if predictions.dim() == 2 and predictions.shape[1] == 1:
                    predictions = predictions.squeeze()
                loss = F.binary_cross_entropy(predictions, y_tensor)
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
                
            return loss
        
        return loss_function


def train_mlp(x: np.ndarray, y: np.ndarray, layer_sizes: List[int], 
              activation: str = 'relu', loss_type: str = 'mse',
              learning_rate: float = 0.01, num_iterations: int = 1000) -> Tuple[np.ndarray, MLP]:
    """
    Train an MLP using the custom gradient descent algorithm.
    
    Args:
        x: Input features (n_samples, input_dim)
        y: Target values (n_samples, output_dim) or (n_samples,)
        layer_sizes: List of layer sizes [input_dim, hidden1, ..., output_dim]
        activation: Activation function ('relu', 'tanh', 'sigmoid')
        loss_type: Loss function type ('mse', 'cross_entropy', 'binary_cross_entropy')
        learning_rate: Learning rate for gradient descent
        num_iterations: Number of training iterations
        
    Returns:
        Tuple of (trained_weights, mlp_model)
    """
    # Validate input dimensions
    if layer_sizes[0] != x.shape[1]:
        raise ValueError(f"Input dimension mismatch: expected {layer_sizes[0]}, got {x.shape[1]}")
    
    # Create MLP
    mlp = MLP(layer_sizes, activation)
    
    # Create gradient descent optimizer
    gradient_descent = GradientDescent(learning_rate=learning_rate, num_iterations=num_iterations)
    gradient_descent.set_weights(mlp.weights)
    
    # Create loss function
    loss_function = mlp.create_loss_function(x, y, loss_type)
    gradient_descent.set_loss_function(loss_function)
    
    # Train the model
    trained_weights = gradient_descent.compute_weight_update()
    
    # Update the MLP with trained weights
    mlp.weights = trained_weights
    
    return trained_weights, mlp


# Example usage and test functions
if __name__ == "__main__":
    # Test with synthetic data
    print("Testing MLP implementation...")
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 100
    input_dim = 3
    X = np.random.randn(n_samples, input_dim)
    # True function: y = sum(X) + noise
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(n_samples)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Define network architecture
    layer_sizes = [input_dim, 5, 3, 1]  # 3 -> 5 -> 3 -> 1
    
    print(f"Network architecture: {layer_sizes}")
    
    # Train the MLP
    print("Training MLP...")
    weights, mlp = train_mlp(X, y, layer_sizes, 
                            activation='relu', 
                            loss_type='mse',
                            learning_rate=0.01, 
                            num_iterations=50)
    
    # Make predictions
    predictions = mlp.predict(X)
    
    # Calculate training error
    mse = np.mean((predictions.flatten() - y) ** 2)
    print(f"Training MSE: {mse:.6f}")
    
    # Test binary classification
    print("\nTesting binary classification...")
    y_binary = (y > np.median(y)).astype(float)
    
    weights_binary, mlp_binary = train_mlp(X, y_binary, [input_dim, 5, 1],
                                          activation='relu',
                                          loss_type='binary_cross_entropy',
                                          learning_rate=0.01,
                                          num_iterations=50)
    
    predictions_binary = mlp_binary.predict(X)
    predictions_binary = 1 / (1 + np.exp(-predictions_binary))  # Apply sigmoid
    accuracy = np.mean((predictions_binary.flatten() > 0.5) == y_binary)
    print(f"Binary classification accuracy: {accuracy:.4f}")
    
    print("MLP implementation completed successfully!")

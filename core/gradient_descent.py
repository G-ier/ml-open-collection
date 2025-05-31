import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, List
import warnings
import torch

class GradientDescent:

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_function = None
        self.tolerance = 1e-6

    def set_loss_function(self, loss_function: Callable):
        self.loss_function = loss_function

    def set_weights(self, weights: np.array):
        self.weights = weights

    def set_tolerance(self, tolerance: float):
        self.tolerance = tolerance

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def set_num_iterations(self, num_iterations: int):
        self.num_iterations = num_iterations
    
    # Singular variable gradient computation
    def compute_gradient_torch(self, grad_func: Callable, w: np.array):
        # Convert numpy array to list for iteration
        w_list = w.tolist() if w.ndim > 0 else [w.item()]
        gradients = []
        
        # Iterate through each element and compute gradient individually
        for i, w_val in enumerate(w_list):
            # Create a scalar tensor for this element
            w_scalar = torch.tensor(w_val, dtype=torch.float32, requires_grad=True)
            y_scalar = grad_func(w_scalar)
            
            # Backward pass for this scalar
            y_scalar.backward()
            
            # Get the gradient for this element
            grad_val = w_scalar.grad.item()
            gradients.append(grad_val)
        
        return np.array(gradients)

    def compute_weight_update_step(self, w: np.array, grad: np.array):

        return w - self.learning_rate * grad
    
    def compute_weight_update(self):

        w = self.weights

        for i in range(self.num_iterations):
            # Compute the gradient
            grad = self.compute_gradient_torch(self.loss_function, w)

            # Compute the weight update
            w = self.compute_weight_update_step(w, grad)

        return w
    

        
"""
if __name__ == "__main__":

    # Test singular functions here

    # Define the gradient function
    def grad_func(x: torch.tensor):
        return x**2

    # Define the gradient descent
    gradient_descent = GradientDescent()

    # Compute the gradient
    gradient = gradient_descent.compute_gradient_torch(grad_func, np.array([1, 2, 3]))
    print("Gradient:", gradient)
    
    # Test a single weight update step
    initial_weights = np.array([1.0, 2.0, 3.0])
    print("Initial weights:", initial_weights)
    
    # Set the loss function for the gradient descent
    gradient_descent.set_loss_function(grad_func)
    
    # Compute gradient at initial weights
    grad_at_weights = gradient_descent.compute_gradient_torch(grad_func, initial_weights)
    print("Gradient at initial weights:", grad_at_weights)
    
    # Perform one weight update step
    updated_weights = gradient_descent.compute_weight_update_step(initial_weights, grad_at_weights)
    print("Updated weights after one step:", updated_weights)
    print("Weight change:", updated_weights - initial_weights)
"""
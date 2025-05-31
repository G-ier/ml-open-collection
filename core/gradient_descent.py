import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, List
import warnings

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

    def compute_gradient(self, x: np.array, y: np.array):
        pass

    def compute_cost(self, x: np.array, y: np.array):
        pass

    def train(self, x: np.array, y: np.array):
        pass
        
        
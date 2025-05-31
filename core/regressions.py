import tqdm
import argparse
import numpy as np

def train_linear_regression(x: np.array, y: np.array):
    """
    Train linear regression model and return the learned weights.
    
    Args:
        x: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
    
    Returns:
        w: Learned weights including bias term
    """

    # Add bias term (column of ones) to the feature matrix
    ones = np.ones((x.shape[0], 1))
    x_with_bias = np.hstack([ones, x])
    
    # Calculate weights using the normal equation: w = (X^T * X)^(-1) * X^T * y
    xt_x = np.dot(x_with_bias.T, x_with_bias)
    xt_x_inv = np.linalg.inv(xt_x)
    xt_y = np.dot(x_with_bias.T, y)
    w = np.dot(xt_x_inv, xt_y)
    
    return w

def linear_regression_function(x: np.array, w: np.array):
    """
    Make predictions using trained linear regression weights.
    
    Args:
        x: Feature matrix (n_samples, n_features)
        w: Trained weights including bias term
    
    Returns:
        predictions: Predicted values
    """
    # Add bias term (column of ones) to the feature matrix
    ones = np.ones((x.shape[0], 1))
    x_with_bias = np.hstack([ones, x])
    
    # Return predictions: y_pred = X * w
    return np.dot(x_with_bias, w)

# Defined as the linear regression of the nonlinear features
def polynomial_regression_function(x: list, d: int, w: np.array):
    
    # Prepare the nonlinear features
    x_nonlinear = np.array([x**i for i in range(0, d + 1)])

    # Train the linear regression
    w = train_linear_regression(x_nonlinear, w)

    # Return the predictions
    return linear_regression_function(x_nonlinear, w)

def logistic_regression_function(x: np.array, w: np.array):
    """
    Make predictions using logistic regression (sigmoid of linear regression).
    
    Args:
        x: Feature matrix (n_samples, n_features)
        w: Trained weights including bias term
    
    Returns:
        predictions: Predicted probabilities (0 to 1)
    """
    # Get linear regression output (w^T * x + b)
    linear_output = linear_regression_function(x, w)
    
    # Apply sigmoid function: σ(z) = 1 / (1 + e^(-z))
    return 1 / (1 + np.exp(-linear_output))

import tqdm
import argparse
import numpy as np
from core.gradient_descent import GradientDescent
from core.data.data_loader import DataLoader
# Linear regression using gradient descent
def train_linear_regression(x: np.array, y: np.array):
    """
    Train linear regression model and return the learned weights.
    
    Args:
        x: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
    
    Returns:
        w: Learned weights including bias term
        f: Learned function
    """
    
    # Define the gradient descent
    gradient_descent = GradientDescent()
    gradient_descent.set_num_iterations(10) # 10 iterations just for testing
    gradient_descent.set_weights(np.zeros(x.shape[1] + 1))
    
    # Define the loss function - MSE loss
    loss_function = lambda w: np.mean((y - linear_regression_function(x, w))**2)
    
    # Set the loss function
    gradient_descent.set_loss_function(loss_function)

    # Compute the weight update
    w = gradient_descent.compute_weight_update()
    
    # Return the learned weights and function with the learned weights
    return w, linear_regression_function(x, w)

def train_polynomial_regression(x: np.array, y: np.array, degree: int):
    """
    Train polynomial regression model and return the learned weights.
    
    Args:
        x: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        degree: Degree of the polynomial
    
    Returns:
        w: Learned weights including bias term
        f: Learned function
    """

    # Add bias term (column of ones) to the feature matrix
    ones = np.ones((x.shape[0], 1))
    x_with_bias = np.hstack([ones, x])
    
    # Define the gradient descent
    gradient_descent = GradientDescent()
    gradient_descent.set_num_iterations(10) # 10 iterations just for testing
    gradient_descent.set_weights(np.zeros(degree + 1)) 
    
    # Define the loss function - Multivariate cross entropy loss
    loss_function = lambda w: np.mean((y - polynomial_regression_function(x_with_bias, degree, w))**2)
    
    # Set the loss function
    gradient_descent.set_loss_function(loss_function)

    # Compute the weight update
    w = gradient_descent.compute_weight_update()
    
    # Return the learned weights and function with the learned weights
    return w, polynomial_regression_function(x, degree, w)

def train_logistic_regression(x: np.array, y: np.array):
    """
    Train logistic regression model and return the learned weights.
    
    Args:
        x: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
    
    Returns:
        w: Learned weights including bias term
        f: Learned function
    """

    # Add bias term (column of ones) to the feature matrix
    ones = np.ones((x.shape[0], 1))
    x_with_bias = np.hstack([ones, x])
    
    # Define the gradient descent
    gradient_descent = GradientDescent()
    gradient_descent.set_num_iterations(10) # 10 iterations just for testing
    gradient_descent.set_weights(np.zeros(x_with_bias.shape[1])) 
    
    # Define the loss function - Binary cross entropy loss
    loss_function = lambda w: -np.mean(y * np.log(logistic_regression_function(x_with_bias, w)) + 
                                      (1 - y) * np.log(1 - logistic_regression_function(x_with_bias, w)))
    
    # Set the loss function
    gradient_descent.set_loss_function(loss_function)

    # Compute the weight update
    w = gradient_descent.compute_weight_update()
    
    # Return the learned weights and function with the learned weights
    return w, logistic_regression_function(x, w)

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
    
    # Prepare the nonlinear features - add first item from x, then polynomial terms
    x_nonlinear = np.array([x[0]] + [x**i for i in range(1, d + 1)])

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

if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    args = parser.parse_args()

    # Define the type of regression
    if args.type == "linear":

        # Load the data
        data_loader = DataLoader()
        x, y = data_loader.load_3d_data()

        # Test the linear regression
        w, f = train_linear_regression(np.array(args.x), np.array(args.y))

        print("Learned weights:", w)
        print("Learned function:", f)

    # Test the linear regression
    w, f = train_linear_regression(np.array(args.x), np.array(args.y))
    print("Learned weights:", w)
    print("Learned function:", f)

    # Test the polynomial regression
    w, f = train_polynomial_regression(np.array([1, 2, 3]), 2, np.array([1, 2, 3]))
    print("Learned weights:", w)
    print("Learned function:", f)

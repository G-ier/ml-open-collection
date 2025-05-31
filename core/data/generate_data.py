import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_linear_data(n_samples=100, n_features=1, noise=0.1, random_state=42):
    """
    Generate synthetic linear regression data
    
    Args:
        n_samples: Number of data points
        n_features: Number of features (input dimensions)
        noise: Standard deviation of noise to add
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target values (n_samples,)
        true_weights: The true weights used to generate the data
        true_bias: The true bias used to generate the data
    """
    np.random.seed(random_state)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate true weights and bias
    true_weights = np.random.randn(n_features) * 2  # Scale weights
    true_bias = np.random.randn() * 0.5
    
    # Generate target values with linear relationship + noise
    y = X @ true_weights + true_bias + np.random.normal(0, noise, n_samples)
    
    return X, y, true_weights, true_bias

def save_data_to_csv(X, y, filename):
    """Save data to CSV file"""
    # Combine features and target into one dataframe
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def plot_data(X, y, filename=None):
    """Plot data if it's 1D"""
    if X.shape[1] == 1:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], y, alpha=0.6)
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Linear Regression Data')
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename)
            print(f"Plot saved to {filename}")
        plt.show()

if __name__ == "__main__":
    # Generate different datasets
    
    # 1. Simple 1D linear regression
    print("Generating 1D linear regression data...")
    X_1d, y_1d, weights_1d, bias_1d = generate_linear_data(n_samples=100, n_features=1, noise=0.2)
    save_data_to_csv(X_1d, y_1d, 'core/data/linear_1d.csv')
    plot_data(X_1d, y_1d, 'core/data/linear_1d_plot.png')
    print(f"True weight: {weights_1d[0]:.3f}, True bias: {bias_1d:.3f}")
    
    # 2. Multi-dimensional linear regression
    print("\nGenerating 3D linear regression data...")
    X_3d, y_3d, weights_3d, bias_3d = generate_linear_data(n_samples=200, n_features=3, noise=0.15)
    save_data_to_csv(X_3d, y_3d, 'core/data/linear_3d.csv')
    print(f"True weights: {weights_3d}")
    print(f"True bias: {bias_3d:.3f}")
    
    # 3. Larger dataset with more noise
    print("\nGenerating larger noisy dataset...")
    X_large, y_large, weights_large, bias_large = generate_linear_data(n_samples=500, n_features=2, noise=0.3)
    save_data_to_csv(X_large, y_large, 'core/data/linear_large_noisy.csv')
    print(f"True weights: {weights_large}")
    print(f"True bias: {bias_large:.3f}")
    
    # 4. Small clean dataset for testing
    print("\nGenerating small clean dataset...")
    X_clean, y_clean, weights_clean, bias_clean = generate_linear_data(n_samples=50, n_features=1, noise=0.05)
    save_data_to_csv(X_clean, y_clean, 'core/data/linear_clean.csv')
    print(f"True weight: {weights_clean[0]:.3f}, True bias: {bias_clean:.3f}")
    
    print("\nAll datasets generated successfully!") 
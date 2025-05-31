import pandas as pd
import numpy as np
from typing import Tuple
import os

class DataLoader:
    """Utility class to load linear regression datasets"""
    
    def __init__(self, data_dir: str = "core/data"):
        self.data_dir = data_dir
        
    def load_dataset(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a dataset from CSV file
        
        Args:
            filename: Name of the CSV file (without path)
            
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset {filepath} not found")
            
        df = pd.read_csv(filepath)
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        target_col = 'target'
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y
    
    def load_1d_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the 1D linear regression dataset"""
        return self.load_dataset('linear_1d.csv')
    
    def load_3d_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the 3D linear regression dataset"""
        return self.load_dataset('linear_3d.csv')
    
    def load_large_noisy_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the large noisy dataset"""
        return self.load_dataset('linear_large_noisy.csv')
    
    def load_clean_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load the small clean dataset"""
        return self.load_dataset('linear_clean.csv')
    
    def list_available_datasets(self):
        """List all available CSV datasets"""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        print("Available datasets:")
        for file in csv_files:
            print(f"  - {file}")
        return csv_files
    
    def get_dataset_info(self, filename: str):
        """Get basic information about a dataset"""
        X, y = self.load_dataset(filename)
        print(f"Dataset: {filename}")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  Target mean: {y.mean():.3f}")
        print(f"  Target std: {y.std():.3f}")

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    print("Testing DataLoader...")
    loader.list_available_datasets()
    
    print("\n" + "="*50)
    
    # Test loading each dataset
    datasets = ['linear_1d.csv', 'linear_3d.csv', 'linear_large_noisy.csv', 'linear_clean.csv']
    
    for dataset in datasets:
        print()
        loader.get_dataset_info(dataset)
        
    print("\n" + "="*50)
    print("Example: Loading 1D data")
    X, y = loader.load_1d_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"First 5 samples:")
    for i in range(5):
        print(f"  X[{i}] = {X[i]}, y[{i}] = {y[i]:.3f}") 
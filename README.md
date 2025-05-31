# ML/NLP Implementations Collection

This repository contains multiple machine learning and natural language processing implementations, each demonstrating different techniques and approaches for various tasks.

## 🚀 Implementations Overview

### 1. DistilBERT Fine-Tuning (`distilbert_finetune.py`)
A complete implementation for fine-tuning DistilBERT on sentiment analysis tasks with frozen base model weights.

**Features:**
- Uses pre-trained DistilBERT (66M parameters) with frozen weights
- Custom classification head for sentiment analysis
- Supports PyTorch MPS for Apple Silicon acceleration
- Trains on SST-2 (Stanford Sentiment Treebank) dataset
- Includes comprehensive training and evaluation loops

**Key Techniques:**
- Transfer learning with frozen base model
- Custom PyTorch Dataset and DataLoader implementation
- Learning rate scheduling with warmup
- Gradient clipping for stable training

### 2. Custom Tokenizer & Embeddings (`custom_tokenizer.py`)
A specialized tokenizer implementation using Jina embeddings with extended context length support.

**Features:**
- Jina embeddings v2 base model integration
- Support for 4500 token context length
- Proper mean pooling and L2 normalization
- Apple Silicon MPS optimization
- Modular design for easy integration

**Key Techniques:**
- Extended context window handling
- Attention mask-aware mean pooling
- L2 normalization for embedding consistency
- Device-agnostic model loading

### 3. Mini RAG System (`mini_rag.py`)
A lightweight Retrieval-Augmented Generation (RAG) pipeline implementation using ChromaDB and sentence transformers.

**Features:**
- PDF document processing and chunking
- Vector database storage with ChromaDB
- Sentence transformer embeddings (E5-base-v2)
- Configurable chunking strategies
- Custom tokenizer integration support
- Adaptive retrieval with model-specific chunk sizes
- Intelligent reranking for improved relevance

**Retrieval Configuration:**
- **Jina AI Embedding Model**: Retrieves 10 chunks with reranking
- **intfloat/e5-base-v2 Model**: Retrieves 20 chunks with reranking
- Both configurations apply reranking to improve result quality and relevance

**Key Techniques:**
- Document chunking with overlap
- Vector similarity search
- Embedding-based retrieval
- Modular pipeline architecture
- Model-adaptive retrieval strategies

### 4. Core ML Algorithms (`core/` folder)
A comprehensive collection of fundamental machine learning algorithms implemented from scratch for educational purposes.

**Implementations:**
- **Gradient Descent** (`gradient_descent.py`): Advanced gradient descent optimizer with PyTorch integration for automatic differentiation
- **Regression Algorithms** (`regressions.py`): Unified implementation containing:
  - **Linear Regression**: Basic linear regression with gradient descent optimization
  - **Polynomial Regression**: Extension of linear regression for non-linear relationships
  - **Logistic Regression**: Binary classification implementation with sigmoid activation

**Gradient Descent Features:**
- **PyTorch Integration**: Automatic differentiation using PyTorch tensors for precise gradient computation
- **Element-wise Processing**: Handles vector inputs by computing gradients for each element individually
- **Configurable Parameters**: Adjustable learning rate, iteration count, and convergence tolerance
- **Weight Update Methods**: Single-step and multi-iteration weight update capabilities
- **Modular Design**: Easy integration with different loss functions and optimization strategies

**Regression Algorithms Features:**
- **Unified Training Interface**: Consistent API across all regression types
- **Automatic Bias Handling**: Automatic addition of bias terms to feature matrices
- **Loss Function Integration**: MSE for linear/polynomial regression, binary cross-entropy for logistic regression
- **Prediction Functions**: Separate prediction functions for each algorithm type
- **Command-line Interface**: Argument parsing for different regression types

**Data Generation & Management:**
- **Synthetic Dataset Generator** (`core/data/generate_data.py`): Creates various linear regression datasets with known ground truth parameters
  - 1D linear regression data (100 samples) with visualization
  - 3D multi-dimensional data (200 samples, 3 features)
  - Large noisy dataset (500 samples, 2 features) for robustness testing
  - Small clean dataset (50 samples) for convergence verification
- **Data Loader Utility** (`core/data/data_loader.py`): Convenient interface for loading generated datasets
- **Ground Truth Tracking**: All datasets include known true weights and biases for algorithm validation

**Features:**
- Pure NumPy implementations for educational clarity
- PyTorch integration for advanced gradient computation
- Comprehensive documentation and comments
- Modular design for easy experimentation and modification
- Synthetic data generation with controllable parameters

**Key Techniques:**
- Automatic differentiation with PyTorch
- Element-wise gradient computation for vector inputs
- Gradient descent optimization with configurable parameters
- Feature scaling and normalization
- Convergence criteria and early stopping
- Synthetic data generation with noise control

## 📋 Requirements

All implementations share common dependencies listed in `requirements.txt`:

```bash
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
numpy>=1.24.0
scikit-learn>=1.2.0
tqdm>=4.65.0
```

Additional dependencies for specific implementations:
- **RAG System**: `chromadb`, `sentence-transformers`, `langchain`, `pypdf`
- **Custom Tokenizer**: `jinaai` models from Hugging Face

## 🛠️ Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install base dependencies
pip install -r requirements.txt

# For RAG system, install additional dependencies:
pip install chromadb sentence-transformers langchain pypdf

# For custom tokenizer with Jina models:
pip install sentence-transformers
```

## 🚀 Usage

### DistilBERT Fine-Tuning

```bash
python distilbert_finetune.py
```

**Configuration:**
- Modify hyperparameters at the top of the script
- Adjust `train_size` and `val_size` for different dataset sizes
- Change `num_labels` in `DistilBERTClassifier` for multi-class tasks

**Output:**
- Trained model saved as `distilbert_classifier.pt`
- Training progress with loss and accuracy metrics

### Custom Tokenizer

```python
from custom_tokenizer import custom_4500_token_encoder

# Encode text with extended context
text = "Your long text here..."
embeddings = custom_4500_token_encoder(text)
```

### Mini RAG System

```python
from mini_rag import rag_pipeline

# Process a PDF and query it
query = "What is the main topic?"
file_path = "path/to/your/document.pdf"
rag_pipeline(query, file_path)
```

### Core ML Algorithms

```python
# Navigate to the core directory
cd core/

# Run regression algorithms with different types
python regressions.py --type linear

# Test the gradient descent implementation
python gradient_descent.py

# Generate synthetic datasets for testing
python data/generate_data.py
```

**Gradient Descent Usage:**
```python
from core.gradient_descent import GradientDescent
from core.data.data_loader import DataLoader
import torch
import numpy as np

# Load data
loader = DataLoader()
X, y = loader.load_1d_data()

# Define a loss function (e.g., mean squared error)
def mse_loss(w):
    # w represents weights, implement your loss function here
    return torch.sum((X @ w - y) ** 2) / len(y)

# Initialize gradient descent
gd = GradientDescent(learning_rate=0.01, num_iterations=1000)
gd.set_loss_function(mse_loss)

# Perform weight updates
initial_weights = np.random.randn(X.shape[1])
gd.set_weights(initial_weights)
final_weights = gd.compute_weight_update()
```

**Regression Algorithms Usage:**
```python
from core.regressions import train_linear_regression, train_polynomial_regression, train_logistic_regression
from core.data.data_loader import DataLoader
import numpy as np

# Load data
loader = DataLoader()
X, y = loader.load_3d_data()

# Train linear regression
weights, predictions = train_linear_regression(X, y)
print("Linear regression weights:", weights)

# Train polynomial regression
poly_weights, poly_predictions = train_polynomial_regression(X[:, 0], y, degree=2)
print("Polynomial regression weights:", poly_weights)

# For binary classification, prepare binary labels
y_binary = (y > np.median(y)).astype(int)
logistic_weights, logistic_predictions = train_logistic_regression(X, y_binary)
print("Logistic regression weights:", logistic_weights)
```

**Data Generation:**
```python
from core.data.data_loader import DataLoader

# Load pre-generated datasets
loader = DataLoader()

# Load different datasets
X_1d, y_1d = loader.load_1d_data()        # Simple 1D regression
X_3d, y_3d = loader.load_3d_data()        # Multi-dimensional
X_large, y_large = loader.load_large_noisy_data()  # Large noisy dataset
X_clean, y_clean = loader.load_clean_data()        # Small clean dataset

# Get dataset information
loader.get_dataset_info('linear_1d.csv')
```

**Configuration:**
- Each algorithm includes configurable parameters at the top of the script
- Modify dataset generation parameters for different problem complexities
- Adjust hyperparameters like learning rate, epochs, and regularization
- Enable/disable visualization for training progress and results

**Output:**
- Training progress with loss curves and accuracy metrics
- Model performance evaluation and comparison metrics

## 📝 Performance Notes

### Hardware Optimization
- **Apple Silicon**: All implementations support MPS acceleration
- **CUDA**: DistilBERT fine-tuning supports GPU acceleration
- **Memory**: RAG system uses efficient chunking for large documents

### Benchmarks
- **DistilBERT**: ~5-10 minutes training on MacBook Air M3 (5K examples)
- **Custom Tokenizer**: Real-time encoding for 4500 tokens
- **RAG System**: Efficient retrieval from large document collections

## 🔧 Customization

### For Your Own Data

**DistilBERT Fine-Tuning:**
1. Replace SST-2 dataset loading with your data
2. Adjust `num_labels` for your classification task
3. Modify the classification head architecture if needed

**RAG System:**
1. Change document loaders for different file types
2. Adjust chunking parameters for your content
3. Swap embedding models for domain-specific tasks

**Custom Tokenizer:**
1. Replace Jina model with your preferred embedding model
2. Adjust `max_length` for different context requirements
3. Modify pooling strategy for specific use cases

**Core ML Algorithms:**
1. Modify dataset generation functions for your specific data
2. Experiment with different optimization algorithms and learning rates
3. Add regularization techniques for improved generalization
4. Implement additional evaluation metrics for your use case

## 📁 Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Base dependencies
├── distilbert_finetune.py   # DistilBERT sentiment analysis
├── custom_tokenizer.py      # Extended context tokenizer
├── mini_rag.py             # RAG pipeline implementation
├── core/                    # Core ML algorithms folder
│   ├── gradient_descent.py     # Advanced gradient descent with PyTorch integration
│   ├── regressions.py          # Unified regression algorithms (linear, polynomial, logistic)
│   └── data/                   # Data generation and management
│       ├── generate_data.py       # Synthetic dataset generator
│       ├── data_loader.py         # Dataset loading utility
│       ├── linear_1d.csv          # 1D linear regression dataset
│       ├── linear_3d.csv          # 3D multi-dimensional dataset
│       ├── linear_large_noisy.csv # Large noisy dataset
│       ├── linear_clean.csv       # Small clean dataset
│       └── linear_1d_plot.png     # Visualization of 1D data
├── db/                      # Database storage for RAG system
└── __init__.py             # Package initialization
```

## 🤝 Contributing

Each implementation is designed to be modular and extensible. Feel free to:
- Add new embedding models to the custom tokenizer
- Extend the RAG system with reranking capabilities
- Implement additional fine-tuning strategies for DistilBERT
- Add support for other transformer architectures
- Enhance core ML algorithms with advanced optimization techniques
- Extend the gradient descent implementation with additional optimizers (Adam, RMSprop, etc.)
- Add new synthetic data generators for different problem types
- Implement additional loss functions and regularization techniques
- Add new fundamental algorithms to the core collection (neural networks, clustering, etc.)

## 📝 Notes

- Models are not included in the repository due to size constraints
- All implementations prioritize educational clarity over production optimization
- Device selection is automatic based on available hardware (MPS > CUDA > CPU)
- The RAG system creates a persistent database in the `db/` directory for efficient retrieval
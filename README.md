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
- **Linear Regression**: Basic linear regression with gradient descent optimization
- **Polynomial Regression**: Extension of linear regression for non-linear relationships
- **Logistic Regression**: Binary and multi-class classification implementation
- **Multi-Layer Perceptron (MLP)**: Neural network with configurable hidden layers
- **K-Means Clustering**: Unsupervised clustering algorithm with centroid optimization

**Features:**
- Pure NumPy implementations for educational clarity
- Comprehensive documentation and comments
- Visualization utilities for understanding algorithm behavior
- Modular design for easy experimentation and modification
- Performance comparison utilities

**Key Techniques:**
- Gradient descent optimization
- Backpropagation for neural networks
- Cross-validation and model evaluation
- Feature scaling and normalization
- Convergence criteria and early stopping

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

# Run individual algorithms
python linear_regression.py
python polynomial_regression.py
python logistic_regression.py
python mlp.py
python kmeans.py
```

**Configuration:**
- Each algorithm includes configurable parameters at the top of the script
- Modify dataset generation parameters for different problem complexities
- Adjust hyperparameters like learning rate, epochs, and regularization
- Enable/disable visualization for training progress and results

**Output:**
- Training progress with loss curves and accuracy metrics
- Visualization plots for algorithm behavior and results
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
│   ├── linear_regression.py    # Linear regression implementation
│   ├── polynomial_regression.py # Polynomial regression implementation
│   ├── logistic_regression.py  # Logistic regression implementation
│   ├── mlp.py                  # Multi-layer perceptron implementation
│   └── kmeans.py              # K-means clustering implementation
└── __init__.py             # Package initialization
```

## 🤝 Contributing

Each implementation is designed to be modular and extensible. Feel free to:
- Add new embedding models to the custom tokenizer
- Extend the RAG system with reranking capabilities
- Implement additional fine-tuning strategies for DistilBERT
- Add support for other transformer architectures
- Enhance core ML algorithms with advanced optimization techniques
- Add new fundamental algorithms to the core collection

## 📝 Notes

- Models are not included in the repository due to size constraints
- All implementations prioritize educational clarity over production optimization
- Device selection is automatic based on available hardware (MPS > CUDA > CPU)
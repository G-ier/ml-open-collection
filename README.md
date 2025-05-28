# DistilBERT Fine-Tuning

This project demonstrates how to fine-tune DistilBERT for sentiment analysis by freezing the base model and only training a custom classification head.

## Overview

This implementation:
- Uses the pre-trained DistilBERT model (66M parameters) with frozen weights
- Adds a custom classification head that is trained on your data
- Leverages PyTorch MPS for Apple Silicon acceleration
- Uses the SST-2 (Stanford Sentiment Treebank) dataset for training

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Other dependencies listed in `requirements.txt`

## Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. **Train the model**:
   ```bash
   python distilbert_finetune.py
   ```
   
   The script will:
   - Download the SST-2 dataset automatically
   - Train for 3 epochs by default
   - Save the best model to `distilbert_classifier.pt`

2. **Adjust for your needs**:
   - Change hyperparameters at the top of the script
   - Modify the training data size (currently using 5000 examples for speed)
   - Add custom data or datasets by modifying the dataset loading section

## Performance

On a MacBook Air M3, training with 5000 examples takes approximately 5-10 minutes using MPS acceleration.

## Customization

To adapt this for your own classification task:
1. Replace the dataset loading with your own data
2. Adjust the number of output classes in `DistilBERTClassifier(num_labels=2)`
3. Modify the classification head architecture if needed 

## Github
Trained model not uploaded
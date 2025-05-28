import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import (
    DistilBertTokenizer, 
    DistilBertModel, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import argparse 

# Check for T4 GPU (CUDA) first, then MPS (Apple Metal), then CPU
# Prioritize T4 GPU for better performance
device = torch.device("cuda" if torch.cuda.is_available() else 
                     ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# Set hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# Load SST-2 dataset (Stanford Sentiment Treebank)
# A simple sentiment analysis dataset with binary labels
dataset = load_dataset("sst2")
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_texts = train_dataset["sentence"]
train_labels = train_dataset["label"]
val_texts = validation_dataset["sentence"]
val_labels = validation_dataset["label"]

# Create smaller subset for demonstration purposes
# This helps with training time on MacBook Air
train_size = 15000  # Change this to use more data
val_size = 1500

train_dataset = SentimentDataset(
    train_texts[:train_size], 
    train_labels[:train_size], 
    tokenizer, 
    MAX_LENGTH
)
val_dataset = SentimentDataset(
    val_texts[:val_size], 
    val_labels[:val_size], 
    tokenizer, 
    MAX_LENGTH
)

# Create data loaders
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=BATCH_SIZE
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# Custom classifier using DistilBERT with frozen weights
class DistilBERTClassifier(torch.nn.Module):
    def __init__(self, num_labels=2):
        super(DistilBERTClassifier, self).__init__()
        # Load the pre-trained DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Freeze all parameters of the base model
        for param in self.distilbert.parameters(): # runs around hundreds of times since thats how many tensors of params there are
            param.requires_grad = False 
            
        # Add custom classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_labels) # logits
        )
        
    def forward(self, input_ids, attention_mask):
        # Get DistilBERT outputs
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the output of [CLS] token (first token)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Apply classification head
        logits = self.classifier(pooled_output)
        
        return logits

# Initialize model
model = DistilBERTClassifier()
model = model.to(device)

# Set up optimizer and scheduler
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    eps=1e-8
)

# Calculate total steps for scheduler
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training loop
def train():
    # Set the model to training mode
    model.train()
    total_loss = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for batch in progress_bar:
        # Get batch tensors and move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Clear gradients
        model.zero_grad() # set all gradients to 0
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate average loss
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

# Evaluation loop
def evaluate():
    # Set the model to evaluation mode
    model.eval()
    all_preds = []
    all_labels = []
    
    # No gradient calculation needed for evaluation
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            # Get batch tensors and move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            
            # Add batch predictions and labels to lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# Example of how to use the model for inference
def predict_sentiment(text):
    # Prepare model for evaluation
    model.eval()
    
    # Tokenize input text
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Move tensors to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    
    return "Positive" if preds[0] == 1 else "Negative"

def main():

    parser = argparse.ArgumentParser(description='Train and evaluate a DistilBERT sentiment classifier')
    parser.add_argument('--train', type=bool, default=False, help='Train model')
    args = parser.parse_args()

    if args.train:
        print("Training model...")
    else:
        print("Skipping to testing model...")

    if args.train:

        # Training and evaluation
        best_accuracy = 0
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            
            # Train
            avg_train_loss = train()
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            # Evaluate
            accuracy = evaluate()
            print(f"Validation Accuracy: {accuracy:.4f}")
            
            # Save model if it's the best so far
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), "distilbert_classifier.pt")
                print(f"Model saved with accuracy: {best_accuracy:.4f}")

        print("\nTraining complete!")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
    
    # Test with some example sentences
    try:
        # Try to load the saved model
        model.load_state_dict(torch.load("distilbert_classifier.pt"))
        
        # Test examples
        examples = [
            "I love this movie!",
            "This was a terrible experience.",
            "The product works as expected.",
            "I'm not sure how I feel about this."
        ]
        
        print("\nTesting the model with examples:")
        for example in examples:
            sentiment = predict_sentiment(example)
            print(f"Text: '{example}'")
            print(f"Sentiment: {sentiment}\n")
    except:
        print("\nModel hasn't been trained yet. Run the script to train the model first.")

if __name__ == "__main__":
    main() 
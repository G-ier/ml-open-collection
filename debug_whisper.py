import torch
import sys
import os

# Add the asr directory to path so we can import WhisperFT
sys.path.append('asr')

from WhisperFT import WhisperFT, AudioDataset, audio_collate_fn
from torch.utils.data import DataLoader

def debug_model():
    print("Creating model...")
    model = WhisperFT(lstm_hidden_size=128, lstm_num_layers=2)
    model.define_ft_heads(2)
    
    print("Loading data...")
    model.load_data(
        "/Users/gier/Downloads/archive/KAGGLE/training_data", 
        "/Users/gier/Downloads/archive/KAGGLE/eval_data", 
        "/Users/gier/Downloads/archive/KAGGLE/test_data"
    )
    
    print("Creating data loader...")
    train_loader = DataLoader(
        model.train_dataset, 
        batch_size=2,  # Use small batch for debugging
        shuffle=True, 
        collate_fn=audio_collate_fn
    )
    
    print("Getting one batch...")
    for audio_batch, labels_batch in train_loader:
        print(f"Audio batch shape: {audio_batch.shape}")
        print(f"Labels batch shape: {labels_batch.shape}")
        print(f"Labels batch dtype: {labels_batch.dtype}")
        print(f"Labels batch values: {labels_batch}")
        
        print("Testing forward pass...")
        outputs = model.forward_batch(audio_batch)
        print(f"Final outputs shape: {outputs.shape}")
        print(f"Final outputs dtype: {outputs.dtype}")
        
        print("Testing loss calculation...")
        criterion = torch.nn.CrossEntropyLoss()
        try:
            loss = criterion(outputs, labels_batch)
            print(f"Loss computed successfully: {loss}")
        except Exception as e:
            print(f"Loss computation failed: {e}")
            print(f"Expected: outputs shape {outputs.shape}, labels shape {labels_batch.shape}")
            
        break  # Only test one batch

if __name__ == "__main__":
    debug_model() 
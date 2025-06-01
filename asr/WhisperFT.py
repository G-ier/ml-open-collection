import torch
import whisper
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import tqdm

class AudioDataset(Dataset):
    """Custom dataset for loading audio files from directories"""
    
    def __init__(self, data_dir):
        """
        Initialize AudioDataset
        
        Args:
            data_dir: Path to directory containing audio files
                     - If subdirectories exist, each subdirectory is treated as a class
                     - If no subdirectories, all files are treated as class 0
        """
        self.data_dir = data_dir
        self.audio_files = []
        self.labels = []
        self.class_to_idx = {}
        
        # Supported audio extensions
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.ffmpeg']
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory {data_dir} does not exist")
        
        # Check if there are subdirectories (each subdirectory = a class)
        subdirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        
        # Single directory with audio files - files' classes are defined by their name containing "original"
        self.class_to_idx = {'default': 0}
        
        for file in os.listdir(data_dir):
            # Skip non-audio files and system files
            if not any(file.lower().endswith(ext) for ext in audio_extensions):
                continue
            if file.startswith('.'):  # Skip hidden files like .DS_Store
                continue
                
            file_path = os.path.join(data_dir, file)
            self.audio_files.append(file_path)
            
            # Assign label based on filename
            if "original" in file.lower():
                self.labels.append(1)
            else:
                self.labels.append(0)
        
        print(f"Audio dataset: {len(self.audio_files)} files")
        print(f"Labels distribution: {len([l for l in self.labels if l == 0])} class 0, {len([l for l in self.labels if l == 1])} class 1")
            
        
        print(f"Total audio files loaded: {len(self.audio_files)}")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Get audio file and label by index
        
        Returns:
            tuple: (audio_array, label) where audio_array is numpy array
        """
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            # Load audio using whisper's audio loading function
            # This automatically resamples to 16kHz and converts to mono
            audio = whisper.load_audio(audio_path)
            return torch.from_numpy(audio).float(), torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return a zero tensor as fallback (30 seconds of silence at 16kHz)
            return torch.zeros(16000 * 30), torch.tensor(label, dtype=torch.long)
    
    def get_class_names(self):
        """Return mapping of class indices to class names"""
        return {idx: name for name, idx in self.class_to_idx.items()}
    
    def get_class_counts(self):
        """Return count of samples per class"""
        from collections import Counter
        return Counter(self.labels)

def audio_collate_fn(batch):
    """
    Custom collate function to handle variable-length audio
    Pads or truncates audio to 30 seconds (480,000 samples at 16kHz)
    """
    # Whisper expects 30 seconds of audio at 16kHz = 480,000 samples
    target_length = 480000
    
    audios = []
    labels = []
    
    for audio, label in batch:
        # Pad or truncate audio to target length
        if len(audio) > target_length:
            # Truncate
            audio = audio[:target_length]
        else:
            # Pad with zeros
            padding = target_length - len(audio)
            audio = torch.cat([audio, torch.zeros(padding)])
        
        audios.append(audio)
        labels.append(label)
    
    # Stack into batch tensors
    audio_batch = torch.stack(audios)
    label_batch = torch.stack(labels)
    
    return audio_batch, label_batch

class WhisperFT:
    def __init__(self, model_name="tiny", lstm_hidden_size=128, lstm_num_layers=2):
        self.model = whisper.load_model(model_name)
        self.epochs = 1
        self.lr = 1e-3
        self.device = torch.device("cpu")  # Use CPU only for simplicity
        
        # Keep Whisper model on CPU due to sparse tensor limitations with MPS
        # Only fine-tuning layers will be moved to MPS/GPU
        
        self.ft_head = None
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]
    
    def load_data(self, train_path, val_path, test_path):
        """Load audio data from directories containing audio files"""
        self.train_dataset = AudioDataset(train_path)
        self.val_dataset = AudioDataset(val_path)
        self.test_dataset = AudioDataset(test_path)
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    # First ran after creating the model class
    def define_ft_heads(self, num_classes):
        # Freeze all base model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get the dimension of the encoder output
        # For Whisper, we typically use the encoder output dimension
        encoder_dim = self.model.dims.n_audio_state
        
        # Create fine-tuning head layers with LSTM
        self.ft_head = nn.Sequential(
            # LSTM to process temporal features
            nn.LSTM(
                input_size=encoder_dim,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                batch_first=True,
                dropout=0.1 if self.lstm_num_layers > 1 else 0,
                bidirectional=True  # Use bidirectional LSTM for better context
            ),
        ).to(self.device)
        
        # Classification head after LSTM
        # Note: bidirectional LSTM doubles the output size
        lstm_output_size = self.lstm_hidden_size * 2  # *2 for bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        ).to(self.device)

    def forward(self, audio):
        # Get encoder features from the frozen Whisper model
        with torch.no_grad():
            # Process audio through Whisper encoder (keep on CPU)
            mel = whisper.log_mel_spectrogram(audio)  # Keep mel on CPU
            
            # Ensure mel has the right shape: [batch_size, n_mels, time]
            # whisper.log_mel_spectrogram returns [n_mels, time], we need to add batch dimension
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)  # Add batch dimension: [1, n_mels, time]
            
            # Convert to dense tensor if sparse
            if mel.is_sparse:
                mel = mel.to_dense()
                
            encoder_output = self.model.encoder(mel)  # Encoder stays on CPU
        
        # encoder_output shape: [batch_size, seq_len, feature_dim]
        # where seq_len is typically ~1500 for 30-second audio
        
        # Pass through LSTM to process temporal information
        lstm_out, (hidden, cell) = self.ft_head[0](encoder_output)
        # lstm_out shape: [batch_size, seq_len, lstm_hidden_size * 2]
        
        # Option 1: Use the final hidden state (last timestep)
        # Take the last hidden state from both directions
        # hidden shape: [num_layers * 2, batch_size, hidden_size]
        final_hidden = hidden[-2:, :, :]  # Get last layer forward and backward
        final_hidden = torch.cat((final_hidden[0], final_hidden[1]), dim=1)
        
        # Use final hidden state approach (Option 1)
        return self.classifier(final_hidden)

    def prepare_dataloaders(self, train_chunks, val_chunks=None, batch_size=8, shuffle=True):
        """
        Create DataLoaders from audio chunks
        
        Args:
            train_chunks: List of chunk dictionaries from preprocess_audio
            val_chunks: Optional validation chunks
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            
        Returns:
            train_loader, val_loader (or just train_loader if no val_chunks)
        """
        train_dataset = AudioChunkDataset(train_chunks)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0,
            collate_fn=audio_collate_fn
        )
        
        if val_chunks:
            val_dataset = AudioChunkDataset(val_chunks)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0,
                collate_fn=audio_collate_fn
            )
            return train_loader, val_loader
        
        return train_loader

    def train_model(self, train_loader, val_loader=None, epochs=10, lr=1e-3, batch_size=8):
        if self.ft_head is None:
            raise ValueError("Fine-tuning head not defined. Call define_ft_heads() first.")
        
        # Create DataLoader if not provided
        if isinstance(train_loader, AudioDataset):
            train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, collate_fn=audio_collate_fn)
        
        # Optimize both LSTM and classifier parameters
        trainable_params = list(self.ft_head.parameters()) + list(self.classifier.parameters())
        optimizer = optim.Adam(trainable_params, lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.ft_head.train()
        self.classifier.train()
        
        # Progress bar for epochs
        epoch_pbar = tqdm.tqdm(range(epochs), desc="Training Progress", unit="epoch")
        
        for epoch in epoch_pbar:
            total_loss = 0
            correct_predictions = 0
            total_samples = 0
            
            # Progress bar for batches
            batch_pbar = tqdm.tqdm(
                enumerate(train_loader), 
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{epochs}",
                unit="batch",
                leave=False
            )
            
            for batch_idx, (audio_batch, labels_batch) in batch_pbar:
                audio_batch = audio_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                # Debug prints
                if batch_idx == 0:  # Only print for first batch
                    print(f"Audio batch shape: {audio_batch.shape}")
                    print(f"Labels batch shape: {labels_batch.shape}")
                    print(f"Labels batch dtype: {labels_batch.dtype}")
                    print(f"Labels batch values: {labels_batch}")
                
                optimizer.zero_grad()
                outputs = self.forward_batch(audio_batch)
                
                # Debug print for outputs
                if batch_idx == 0:
                    print(f"Model outputs shape: {outputs.shape}")
                    print(f"Model outputs dtype: {outputs.dtype}")
                    print(f"About to compute loss...")
                    print(f"Outputs for loss: {outputs}")
                    print(f"Labels for loss: {labels_batch}")
                    
                loss = criterion(outputs, labels_batch)
                loss.backward()
                
                # Gradient clipping for LSTM stability
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels_batch.size(0)
                correct_predictions += (predicted == labels_batch).sum().item()
                
                # Update batch progress bar with current loss and accuracy
                batch_accuracy = 100 * correct_predictions / total_samples
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{batch_accuracy:.2f}%'
                })
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct_predictions / total_samples
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Avg Loss': f'{avg_loss:.4f}',
                'Train Acc': f'{accuracy:.2f}%'
            })
            
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
            
            # Validation
            if val_loader:
                self.validate(val_loader, criterion)

    def forward_batch(self, audio_batch):
        """
        Forward pass for a batch of audio
        
        Args:
            audio_batch: Tensor of shape [batch_size, audio_length]
        """
        batch_size = audio_batch.size(0)
        outputs = []
        
        print(f"Processing batch of size: {batch_size}")
        
        for i in range(batch_size):
            # Process each audio in the batch
            audio = audio_batch[i].cpu().numpy()
            output = self.forward(audio)
            print(f"Single forward output shape: {output.shape}")
            # Squeeze out the batch dimension from individual forward pass
            output = output.squeeze(0)  # Remove the batch dimension [1, num_classes] -> [num_classes]
            outputs.append(output)
        
        stacked_outputs = torch.stack(outputs)
        print(f"Stacked outputs shape: {stacked_outputs.shape}")
        return stacked_outputs

    def validate(self, val_loader, criterion):
        """Validation function"""
        self.ft_head.eval()
        self.classifier.eval()
        
        total_val_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            # Progress bar for validation batches
            val_pbar = tqdm.tqdm(
                val_loader,
                desc="Validating",
                unit="batch",
                leave=False
            )
            
            for audio_batch, labels_batch in val_pbar:
                audio_batch = audio_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                outputs = self.forward_batch(audio_batch)
                loss = criterion(outputs, labels_batch)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels_batch.size(0)
                correct_predictions += (predicted == labels_batch).sum().item()
                
                # Update validation progress bar
                val_accuracy = 100 * correct_predictions / total_samples
                val_pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Val Acc': f'{val_accuracy:.2f}%'
                })
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # Set back to training mode
        self.ft_head.train()
        self.classifier.train()

    def test(self, test_loader, criterion=None):
        """Test function"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.ft_head.eval()
        self.classifier.eval()
        
        total_test_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            # Progress bar for test batches
            test_pbar = tqdm.tqdm(
                test_loader,
                desc="Testing",
                unit="batch",
                leave=False
            )
            
            for audio_batch, labels_batch in test_pbar:
                audio_batch = audio_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                
                outputs = self.forward_batch(audio_batch)
                loss = criterion(outputs, labels_batch)
                total_test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels_batch.size(0)
                correct_predictions += (predicted == labels_batch).sum().item()
                
                # Update test progress bar
                test_accuracy = 100 * correct_predictions / total_samples
                test_pbar.set_postfix({
                    'Test Loss': f'{loss.item():.4f}',
                    'Test Acc': f'{test_accuracy:.2f}%'
                })
        
        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = 100 * correct_predictions / total_samples
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        return test_accuracy, avg_test_loss

if __name__ == "__main__":
    model = WhisperFT(lstm_hidden_size=128, lstm_num_layers=2)
    model.define_ft_heads(2)
    model.load_data("/Users/gier/Downloads/archive/KAGGLE/training_data", "/Users/gier/Downloads/archive/KAGGLE/eval_data", "/Users/gier/Downloads/archive/KAGGLE/test_data")
    
    # Create DataLoaders
    train_loader = DataLoader(model.train_dataset, batch_size=8, shuffle=True, collate_fn=audio_collate_fn)
    val_loader = DataLoader(model.val_dataset, batch_size=8, shuffle=False, collate_fn=audio_collate_fn)
    test_loader = DataLoader(model.test_dataset, batch_size=8, shuffle=False, collate_fn=audio_collate_fn)
    
    # Train the model with eval
    model.train_model(train_loader, val_loader, epochs=1, lr=1e-3)

    # Test the model
    model.test(test_loader)
    

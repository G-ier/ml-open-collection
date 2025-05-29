import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def get_device():
    """Get the appropriate device for the model."""
    device = torch.device("mps")
    if device.type == "mps":
        print("Using MPS device")
        return device
    else:
        raise ValueError("Apple metal is not available")

def load_jina_model():
    """Load the Jina embeddings model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-de")
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-de")
    
    # Port the model to the device
    device = get_device()
    model.to(device)
    
    return tokenizer, model, device

# Initialize the model components when the module is imported
tokenizer, model, device = load_jina_model()

def custom_4500_token_encoder(text: str):
    """Encode the text using the Jina model with proper mean pooling and L2 normalization."""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4500)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]
        attention_mask = inputs["attention_mask"]
    
    # Mean pooling
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # unsqueeze adds dimension in the end and this allows for elementwise multiplication while ignoring padded tokens
    
    # Sum the embeddings for each token
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    pooled = sum_embeddings / sum_mask

    # L2 normalization
    normalized_embeddings = F.normalize(pooled, p=2, dim=1)

    return normalized_embeddings

# Export these for easy importing
__all__ = ['tokenizer', 'model', 'device', 'load_jina_model', 'get_device', 'custom_4500_token_encoder']
"""
Default configurations and settings for the vision modules.
"""
from torchvision import transforms

# Default data transformations
DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ImageNet normalization transforms
IMAGENET_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 5,
    'hidden_units': 10
}

# ViT configuration
VIT_CONFIG = {
    'img_size': 224,
    'patch_size': 16,
    'embedding_dim': 768,
    'num_transformer_layers': 12,
    'num_heads': 12,
    'mlp_size': 3072,
    'num_classes': 3  # pizza, steak, sushi
}

# Device configuration
def get_device():
    """Get the available device (CUDA, MPS, or CPU)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
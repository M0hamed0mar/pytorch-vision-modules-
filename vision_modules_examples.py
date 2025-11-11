"""
VISION MODULES USAGE GUIDE & EXAMPLES
=====================================

This file contains comprehensive examples of how to use all the vision modules.
Copy and paste the code snippets you need into your projects.

SETUP IN GOOGLE COLAB:
----------------------
# Install from GitHub
!git clone https://github.com/yourusername/your-repo-name.git
%cd your-repo-name

# Or install directly from files
!wget https://raw.githubusercontent.com/yourusername/your-repo-name/main/data_setup.py
!wget https://raw.githubusercontent.com/yourusername/your-repo-name/main/train_engine.py
!wget https://raw.githubusercontent.com/yourusername/your-repo-name/main/model_utils.py
!wget https://raw.githubusercontent.com/yourusername/your-repo-name/main/vit_model.py
!wget https://raw.githubusercontent.com/yourusername/your-repo-name/main/config.py

# Import the modules
from data_setup import DataDownloader, DataLoaderCreator, download_data, walk_through_dir
from train_engine import Trainer, TrainingTracker, set_seeds
from model_utils import ModelSaver, Predictor
from vit_model import ViT, PatchEmbedding, TransformerEncoderBlock
from config import get_device, IMAGENET_TRANSFORMS, VIT_CONFIG
"""

# =============================================================================
# 1. DATA SETUP MODULE - data_setup.py
# =============================================================================

"""
DATA DOWNLOADER USAGE:
----------------------
Used for downloading and extracting datasets from URLs.
"""

# Example: Download and extract a dataset
"""
from data_setup import DataDownloader

# Download and extract in one step
data_path = DataDownloader.download_and_extract(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="data",
    extract_to="data/pizza_steak_sushi",
    remove_source=True  # Remove zip file after extraction
)

# Or do it step by step
downloaded_file = DataDownloader.download_data(
    source="https://example.com/data.zip",
    destination="data",
    force_download=False  # Set to True to re-download
)

extracted_path = DataDownloader.extract_data(
    file_path=downloaded_file,
    extract_to="data/extracted",
    remove_source=True
)
"""

"""
DATA LOADER CREATOR USAGE:
--------------------------
Used for creating PyTorch DataLoaders from image folders.
"""

# Example: Create DataLoaders for training
"""
from data_setup import DataLoaderCreator
from torchvision import transforms

# Define your transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create DataLoaders
train_dataloader, test_dataloader, class_names = DataLoaderCreator.create_dataloaders(
    train_dir="data/pizza_steak_sushi/train",
    test_dir="data/pizza_steak_sushi/test", 
    transform=data_transforms,
    batch_size=32,
    num_workers=2  # Optional: number of workers for loading
)

print(f"Classes: {class_names}")
print(f"Training batches: {len(train_dataloader)}")
print(f"Test batches: {len(test_dataloader)}")
"""

# Example: Get all image paths from a directory
"""
image_paths = DataLoaderCreator.get_image_paths(
    data_directory="data/pizza_steak_sushi/train",
    extensions=('.jpg', '.jpeg', '.png')  # Optional: specify file extensions
)
print(f"Found {len(image_paths)} images")
"""

# Example: Walk through directory structure
"""
from data_setup import walk_through_dir

walk_through_dir("data/pizza_steak_sushi")
# Output:
# There are 2 directories and 0 images in 'data/pizza_steak_sushi'.
# There are 3 directories and 0 images in 'data/pizza_steak_sushi/train'.
# There are 0 directories and 75 images in 'data/pizza_steak_sushi/train/pizza'.
"""

# Example: Simple download function
"""
from data_setup import download_data

data_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi"
)
print(f"Data downloaded to: {data_path}")
"""

# =============================================================================
# 2. TRAINING ENGINE MODULE - train_engine.py  
# =============================================================================

"""
TRAINER USAGE:
--------------
Used for training and testing PyTorch models.
"""

# Example: Full training loop
"""
from train_engine import Trainer, set_seeds
import torch
import torch.nn as nn

# Set random seeds for reproducibility
set_seeds(42)

# Setup model, loss function, and optimizer
model = YourModel()  # Your custom model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Train the model
results = Trainer.train(
    model=model,
    train_dataloader=train_dataloader,  # From DataLoaderCreator
    test_dataloader=test_dataloader,    # From DataLoaderCreator  
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
    writer=None  # Optional: TensorBoard writer
)

# Results dictionary contains:
# {
#     "train_loss": [list of training losses per epoch],
#     "train_acc": [list of training accuracies per epoch], 
#     "test_loss": [list of test losses per epoch],
#     "test_acc": [list of test accuracies per epoch]
# }
"""

# Example: Individual training and test steps (for custom training loops)
"""
# Single training step
train_loss, train_acc = Trainer.train_step(
    model=model,
    dataloader=train_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer, 
    device=device
)

# Single test step  
test_loss, test_acc = Trainer.test_step(
    model=model,
    dataloader=test_dataloader,
    loss_fn=loss_fn,
    device=device
)
"""

"""
TRAINING TRACKER USAGE:
-----------------------
Used for tracking training progress and visualization.
"""

# Example: TensorBoard logging
"""
from train_engine import TrainingTracker

# Create TensorBoard writer
writer = TrainingTracker.create_writer(
    experiment_name="pizza_steak_sushi_experiment",
    model_name="vit_model", 
    extra="run_1",  # Optional: additional identifier
    base_dir="runs"  # Optional: base directory for logs
)

# Use in training (writer will be passed to Trainer.train())
results = Trainer.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,  
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
    writer=writer  # Pass the writer here
)
"""

# Example: Plot loss curves
"""
# After training, plot the results
TrainingTracker.plot_loss_curves(results)  # results from Trainer.train()
"""

# =============================================================================
# 3. MODEL UTILS MODULE - model_utils.py
# =============================================================================

"""
MODEL SAVER USAGE:
------------------
Used for saving and loading PyTorch models.
"""

# Example: Save a trained model
"""
from model_utils import ModelSaver

ModelSaver.save_model(
    model=trained_model,
    target_dir="models",
    model_name="my_trained_model.pth"  # Must end with .pth or .pt
)
"""

# Example: Load a saved model
"""
from model_utils import ModelSaver

# First create a model with the same architecture
model = YourModelClass()

# Then load the saved weights
model = ModelSaver.load_model(
    model=model,
    model_path="models/my_trained_model.pth"
)
"""

"""
PREDICTOR USAGE:
----------------
Used for making predictions and visualization.
"""

# Example: Predict and plot an image
"""
from model_utils import Predictor
from torchvision import transforms

Predictor.pred_and_plot_image(
    model=trained_model,
    class_names=["pizza", "steak", "sushi"],  # Your class names
    image_path="path/to/your/image.jpg",
    image_size=(224, 224),  # Optional: resize image
    transform=None,  # Optional: custom transforms
    device="cuda"  # Optional: specify device
)
"""

# Example: Predict without plotting (just get results)
"""
predicted_class, confidence = Predictor.predict_single_image(
    model=trained_model,
    image_path="path/to/your/image.jpg", 
    class_names=["pizza", "steak", "sushi"],
    transform=None,  # Optional: custom transforms
    device="cuda"  # Optional: specify device
)

print(f"Predicted: {predicted_class} with {confidence:.3f} confidence")
"""

# =============================================================================
# 4. VISION TRANSFORMER MODULE - vit_model.py
# =============================================================================

"""
VISION TRANSFORMER USAGE:
------------------------
Complete Vision Transformer model and components.
"""

# Example: Create a complete ViT model
"""
from vit_model import ViT

# Create ViT with default parameters (ViT-Base configuration)
vit_model = ViT(
    img_size=224,                    # Input image size
    in_channels=3,                   # RGB channels
    patch_size=16,                   # Size of each patch
    num_transformer_layers=12,       # Number of transformer blocks
    embedding_dim=768,               # Embedding dimension
    mlp_size=3072,                   # MLP hidden size
    num_heads=12,                    # Number of attention heads
    attn_dropout=0,                  # Attention dropout
    mlp_dropout=0.1,                 # MLP dropout  
    embedding_dropout=0.1,           # Embedding dropout
    num_classes=1000                 # Number of output classes
)

# For custom dataset with 3 classes
custom_vit = ViT(
    img_size=224,
    patch_size=16, 
    num_transformer_layers=4,  # Fewer layers for faster training
    embedding_dim=128,         # Smaller embedding dimension
    num_classes=3              # pizza, steak, sushi
)
"""

# Example: Use individual ViT components
"""
from vit_model import PatchEmbedding, TransformerEncoderBlock

# Patch embedding only
patch_embed = PatchEmbedding(
    in_channels=3,
    patch_size=16, 
    embedding_dim=768
)

# Single transformer block
transformer_block = TransformerEncoderBlock(
    embedding_dim=768,
    num_heads=12,
    mlp_size=3072,
    mlp_dropout=0.1
)

# Forward pass through components
# input_tensor = torch.randn(1, 3, 224, 224)
# patches = patch_embed(input_tensor)          # Shape: [1, 196, 768]
# output = transformer_block(patches)         # Shape: [1, 196, 768]
"""

# =============================================================================
# 5. CONFIG MODULE - config.py  
# =============================================================================

"""
CONFIGURATION USAGE:
--------------------
Pre-defined configurations and utilities.
"""

# Example: Using pre-defined configurations
"""
from config import get_device, IMAGENET_TRANSFORMS, VIT_CONFIG, TRAINING_CONFIG

# Get the best available device
device = get_device()  # Returns "cuda", "mps", or "cpu"

# Use pre-defined transforms
train_dataloader, test_dataloader, class_names = DataLoaderCreator.create_dataloaders(
    train_dir="data/train",
    test_dir="data/test",
    transform=IMAGENET_TRANSFORMS  # Pre-defined ImageNet normalization
)

# Use pre-defined ViT configuration
vit_config = VIT_CONFIG.copy()
vit_config['num_classes'] = len(class_names)  # Adjust for your dataset

model = ViT(**vit_config)

# Use pre-defined training configuration
training_config = TRAINING_CONFIG.copy()
training_config['num_epochs'] = 20  # Adjust as needed
"""

# =============================================================================
# 6. COMPLETE WORKFLOW EXAMPLE
# =============================================================================

"""
COMPLETE TRAINING WORKFLOW:
---------------------------
This shows how to use all modules together in a typical workflow.
"""

# Example: Complete training pipeline
"""
# 1. Imports
from data_setup import DataDownloader, DataLoaderCreator, download_data, walk_through_dir
from train_engine import Trainer, TrainingTracker, set_seeds
from model_utils import ModelSaver, Predictor
from vit_model import ViT
from config import get_device, IMAGENET_TRANSFORMS
import torch
import torch.nn as nn

# 2. Setup seeds and device
set_seeds(42)
device = get_device()
print(f"Using device: {device}")

# 3. Download and prepare data
data_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi"
)

# 4. Explore the data directory
walk_through_dir(data_path)

# 5. Create DataLoaders
train_dataloader, test_dataloader, class_names = DataLoaderCreator.create_dataloaders(
    train_dir=data_path / "train",
    test_dir=data_path / "test",
    transform=IMAGENET_TRANSFORMS,
    batch_size=32
)

print(f"Classes: {class_names}")

# 6. Create model
model = ViT(
    img_size=224,
    patch_size=16,
    num_transformer_layers=4,
    embedding_dim=128, 
    num_classes=len(class_names)
).to(device)

# 7. Setup training
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 8. Train model
results = Trainer.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader, 
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device
)

# 9. Plot results
TrainingTracker.plot_loss_curves(results)

# 10. Save model
ModelSaver.save_model(
    model=model,
    target_dir="models",
    model_name="trained_vit_model.pth"
)

# 11. Make predictions
# Get a sample image
image_paths = DataLoaderCreator.get_image_paths(data_path / "train")
if image_paths:
    sample_image = str(image_paths[0])
    
    # Predict and plot
    Predictor.pred_and_plot_image(
        model=model,
        class_names=class_names,
        image_path=sample_image
    )
    
    # Or just get prediction
    predicted_class, confidence = Predictor.predict_single_image(
        model=model,
        image_path=sample_image,
        class_names=class_names
    )
    print(f"Predicted: {predicted_class} with {confidence:.3f} confidence")
"""

# =============================================================================
# 7. ADVANCED USAGE EXAMPLES
# =============================================================================

"""
ADVANCED USAGE:
---------------
More complex examples showing advanced features.
"""

# Example: Custom training with TensorBoard
"""
from train_engine import Trainer, TrainingTracker, set_seeds
from model_utils import ModelSaver

# Set seeds for reproducibility
set_seeds(42)

# Create TensorBoard writer
writer = TrainingTracker.create_writer(
    experiment_name="advanced_experiment",
    model_name="custom_vit",
    extra="with_augmentation"
)

# Train with TensorBoard logging
results = Trainer.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=15,
    device=device,
    writer=writer  # Enable TensorBoard logging
)

# Save model with timestamp
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ModelSaver.save_model(
    model=model,
    target_dir="models",
    model_name=f"vit_model_{timestamp}.pth"
)
"""

# Example: Using different download methods
"""
from data_setup import DataDownloader, download_data

# Method 1: Simple download (from helper_functions)
simple_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="simple_download"
)

# Method 2: Advanced download with progress bar
advanced_path = DataDownloader.download_and_extract(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="advanced_download",
    extract_to="advanced_download/extracted",
    remove_source=False  # Keep the zip file
)

print(f"Simple download: {simple_path}")
print(f"Advanced download: {advanced_path}")
"""

print("Vision Modules Usage Guide Ready!")
print("All modules are properly integrated")
print("Added set_seeds() for reproducibility") 
print("Added walk_through_dir() for data exploration")
print("Added download_data() for simple downloads")
print("Updated examples with all new functions")
print("\nCopy the code snippets you need and modify them for your projects!")
## **`README.md`**

```markdown
# PyTorch Vision Modules

A collection of modular PyTorch components for computer vision projects, including Vision Transformers, data utilities, training engines, and model management tools.

## 📁 Project Structure

```
pytorch-learning/
├── data_setup.py          # Data downloading, extraction, and DataLoader creation
├── train_engine.py        # Training loops, evaluation, and progress tracking
├── model_utils.py         # Model saving/loading and prediction utilities
├── vit_model.py           # Vision Transformer implementation and components
├── config.py              # Default configurations and settings
├── helper_functions.py    # Additional utility functions
└── going_modular.py       # Modular training scripts
```

## 🚀 Quick Start

### Google Colab Setup
```python
# Clone the repository
!git clone https://github.com/M0hamed0mar/pytorch-learning.git
%cd pytorch-learning

# Import the modules
from data_setup import DataDownloader, DataLoaderCreator, download_data, walk_through_dir
from train_engine import Trainer, TrainingTracker, set_seeds
from model_utils import ModelSaver, Predictor
from vit_model import ViT
from config import get_device, IMAGENET_TRANSFORMS
```

### Basic Usage Example
```python
# 1. Download data
from data_setup import download_data
data_path = download_data(
    source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination="pizza_steak_sushi"
)

# 2. Create DataLoaders
from data_setup import DataLoaderCreator
train_dataloader, test_dataloader, class_names = DataLoaderCreator.create_dataloaders(
    train_dir=data_path / "train",
    test_dir=data_path / "test",
    transform=IMAGENET_TRANSFORMS,
    batch_size=32
)

# 3. Create and train Vision Transformer
from vit_model import ViT
from train_engine import Trainer
import torch.nn as nn

model = ViT(
    img_size=224,
    patch_size=16,
    num_transformer_layers=4,
    embedding_dim=128,
    num_classes=len(class_names)
)

# Train the model
results = Trainer.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=nn.CrossEntropyLoss(),
    epochs=10,
    device=get_device()
)
```

## 📚 Module Documentation

### data_setup.py
**Data preparation and loading utilities**

- `DataDownloader` - Download and extract datasets with progress bars
- `DataLoaderCreator` - Create PyTorch DataLoaders from image folders
- `download_data()` - Simple one-line dataset download
- `walk_through_dir()` - Explore dataset structure

### train_engine.py  
**Training and evaluation engine**

- `Trainer` - Complete training pipeline with metrics tracking
- `TrainingTracker` - TensorBoard logging and visualization
- `set_seeds()` - Reproducibility utility

### model_utils.py
**Model management and inference**

- `ModelSaver` - Save and load model weights
- `Predictor` - Make predictions and visualize results

### vit_model.py
**Vision Transformer implementation**

- `ViT` - Complete Vision Transformer model
- `PatchEmbedding` - Convert images to patch sequences
- `TransformerEncoderBlock` - Transformer layers with residual connections

### config.py
**Default configurations**

- Pre-defined transforms (`IMAGENET_TRANSFORMS`)
- ViT configurations (`VIT_CONFIG`)
- Training settings (`TRAINING_CONFIG`)
- Device detection (`get_device()`)

## 🛠 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
requests>=2.25.0
tqdm>=4.60.0
```
```
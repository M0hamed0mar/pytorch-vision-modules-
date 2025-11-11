# PyTorch Vision Modules

## Overview

This repository contains a collection of modular PyTorch components for computer vision tasks.  
It includes Vision Transformer implementations, training engines, model management utilities, and data handling scripts.  
The repository is designed for modularity and reusability, allowing users to integrate and experiment with different components independently.

---

## Structure

```

pytorch-learning/
├── data_setup.py          # Handles data downloading, extraction, and DataLoader creation
├── train_engine.py        # Contains training and evaluation pipeline
├── model_utils.py         # Utilities for saving, loading, and making predictions
├── vit_model.py           # Vision Transformer implementation and its components
├── config.py              # Default configurations, transforms, and device settings
├── helper_functions.py    # General utility functions
└── going_modular.py       # Example script showing modular usage of components

````

---

## Quick Start

### Using the Repository

1. **Clone the repository locally**
```bash
git clone https://github.com/<your-username>/pytorch-learning.git
cd pytorch-learning
````

2. **Push local changes to GitHub**

```bash
git add .
git commit -m "Your commit message"
git push origin main
```

3. **Access the repository from Google Colab**

* In Colab, you can clone the repository directly to access all modules:

```python
!git clone https://github.com/<your-username>/pytorch-learning.git
%cd pytorch-learning
```

* After this step, all scripts and modules in the repository are available for import and usage in Colab notebooks.

---

## Modules Overview

* `data_setup.py`: Data downloading, extraction, and DataLoader utilities
* `train_engine.py`: Training loop, evaluation, and metrics tracking
* `model_utils.py`: Model saving/loading and prediction utilities
* `vit_model.py`: Vision Transformer model implementation
* `config.py`: Default configurations including transforms and device setup
* `helper_functions.py`: Miscellaneous helper functions
* `going_modular.py`: Example of combining all components for modular usage


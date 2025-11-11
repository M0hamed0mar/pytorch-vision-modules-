"""
Utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


class ModelSaver:
    """Handles saving and loading PyTorch models."""
    
    @staticmethod
    def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
        """
        Save a PyTorch model to a target directory.
        
        Args:
            model: A target PyTorch model to save
            target_dir: A directory for saving the model to
            model_name: A filename for the saved model
            
        Example usage:
            save_model(model=model_0,
                      target_dir="models",
                      model_name="tinyvgg_model.pth")
        """
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True, exist_ok=True)
        
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_save_path = target_dir_path / model_name
        
        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)
    
    @staticmethod
    def load_model(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
        """
        Load a PyTorch model from a file.
        
        Args:
            model: Model architecture to load weights into
            model_path: Path to the saved model weights
            
        Returns:
            Model with loaded weights
        """
        model.load_state_dict(torch.load(model_path))
        print(f"[INFO] Loaded model from: {model_path}")
        return model


class Predictor:
    """Handles making predictions with trained models."""
    
    @staticmethod
    def pred_and_plot_image(model: torch.nn.Module, class_names: List[str], image_path: str,
                           image_size: Tuple[int, int] = (224, 224), transform: transforms.Compose = None,
                           device: torch.device = None):
        """
        Predict on a target image and plot the result.
        
        Args:
            model: A trained PyTorch model
            class_names: List of target classes
            image_path: Filepath to target image
            image_size: Size to transform target image to
            transform: Transform to perform on image
            device: Target device to perform prediction on
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Open image
        img = Image.open(image_path)
        
        # Create transformation for image
        if transform is not None:
            image_transform = transform
        else:
            image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Make prediction
        model.to(device)
        model.eval()
        with torch.inference_mode():
            transformed_image = image_transform(img).unsqueeze(dim=0)
            target_image_pred = model(transformed_image.to(device))
        
        # Convert logits to probabilities
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        
        # Plot image with predicted label and probability
        plt.figure()
        plt.imshow(img)
        plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
        plt.axis(False)
        plt.show()
    
    @staticmethod
    def predict_single_image(model: torch.nn.Module, image_path: str, class_names: List[str],
                           transform: transforms.Compose = None, device: torch.device = None) -> Tuple[str, float]:
        """
        Predict on a single image and return the result.
        
        Args:
            model: A trained PyTorch model
            image_path: Filepath to target image
            class_names: List of target classes
            transform: Transform to perform on image
            device: Target device to perform prediction on
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Open and transform image
        img = Image.open(image_path)
        
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Make prediction
        model.to(device)
        model.eval()
        with torch.inference_mode():
            transformed_image = transform(img).unsqueeze(0).to(device)
            prediction = model(transformed_image)
            probabilities = torch.softmax(prediction, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities.max().item()
        
        return class_names[predicted_class_idx], confidence
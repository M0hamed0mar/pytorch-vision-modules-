"""
Training and evaluation engine for PyTorch models.
"""
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import matplotlib.pyplot as plt


def set_seeds(seed: int = 42):
    """
    Sets random seeds for torch operations.
    
    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Trainer:
    """Handles training and testing of PyTorch models."""
    
    @staticmethod
    def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                  loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
                  device: torch.device) -> Tuple[float, float]:
        """
        Train a PyTorch model for a single epoch.
        
        Args:
            model: A PyTorch model to be trained
            dataloader: A DataLoader instance for training data
            loss_fn: A PyTorch loss function
            optimizer: A PyTorch optimizer
            device: Target device to compute on
            
        Returns:
            Tuple of (train_loss, train_accuracy)
        """
        model.train()
        train_loss, train_acc = 0, 0
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
            
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc
    
    @staticmethod
    def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                 loss_fn: torch.nn.Module, device: torch.device) -> Tuple[float, float]:
        """
        Test a PyTorch model for a single epoch.
        
        Args:
            model: A PyTorch model to be tested
            dataloader: A DataLoader instance for testing data
            loss_fn: A PyTorch loss function
            device: Target device to compute on
            
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        model.eval()
        test_loss, test_acc = 0, 0
        
        with torch.inference_mode():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                
                test_pred_logits = model(X)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
                
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc
    
    @staticmethod
    def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, 
              test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.Module, epochs: int, device: torch.device,
              writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None) -> Dict[str, List]:
        """
        Train and test a PyTorch model.
        
        Args:
            model: A PyTorch model to be trained and tested
            train_dataloader: DataLoader instance for training data
            test_dataloader: DataLoader instance for testing data
            optimizer: PyTorch optimizer
            loss_fn: PyTorch loss function
            epochs: Number of epochs to train for
            device: Target device to compute on
            writer: A SummaryWriter instance for logging
            
        Returns:
            Dictionary of training and testing metrics
        """
        results = {
            "train_loss": [],
            "train_acc": [], 
            "test_loss": [],
            "test_acc": []
        }
        
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = Trainer.train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn, 
                optimizer=optimizer,
                device=device
            )
            
            test_loss, test_acc = Trainer.test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device
            )
            
            print(
                f"Epoch: {epoch+1}/{epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )
            
            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            
            # Log to TensorBoard if writer is provided
            if writer:
                writer.add_scalars(
                    main_tag="Loss", 
                    tag_scalar_dict={
                        "train_loss": train_loss,
                        "test_loss": test_loss
                    },
                    global_step=epoch
                )
                writer.add_scalars(
                    main_tag="Accuracy", 
                    tag_scalar_dict={
                        "train_acc": train_acc,
                        "test_acc": test_acc
                    }, 
                    global_step=epoch
                )
                writer.flush()
        
        if writer:
            writer.close()
            print(f"[INFO] Training completed. TensorBoard logs saved.")
            
        return results


class TrainingTracker:
    """Handles training progress tracking and visualization."""
    
    @staticmethod
    def create_writer(experiment_name: str, model_name: str, extra: str = None,
                     base_dir: str = "runs") -> torch.utils.tensorboard.writer.SummaryWriter:
        """
        Create a TensorBoard SummaryWriter instance.
        
        Args:
            experiment_name: Name of experiment
            model_name: Name of model
            extra: Anything extra to add to the directory
            base_dir: Base directory for runs
            
        Returns:
            SummaryWriter instance
        """
        from torch.utils.tensorboard import SummaryWriter
        
        timestamp = datetime.now().strftime("%Y-%m-%d")
        log_dir_parts = [base_dir, timestamp, experiment_name, model_name]
        if extra:
            log_dir_parts.append(extra)
        
        log_dir = os.path.join(*log_dir_parts)
            
        print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")
        return SummaryWriter(log_dir=log_dir)
    
    @staticmethod
    def plot_loss_curves(results: Dict[str, List]):
        """
        Plot training and testing loss and accuracy curves.
        
        Args:
            results: Dictionary containing training and testing metrics
        """
        # Get the loss values of the results dictionary
        loss = results["train_loss"]
        test_loss = results["test_loss"]
        
        # Get the accuracy values of the results dictionary
        accuracy = results["train_acc"]
        test_accuracy = results["test_acc"]
        
        # Figure out how many epochs there were
        epochs = range(len(results["train_loss"]))
        
        # Setup a plot
        plt.figure(figsize=(15, 7))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label="train_loss")
        plt.plot(epochs, test_loss, label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label="train_accuracy")
        plt.plot(epochs, test_accuracy, label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        
        plt.show()
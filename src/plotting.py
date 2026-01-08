import matplotlib.pyplot as plt
import wandb
import os
from typing import List, Tuple
import numpy as np

def plot_training_results(
    train_losses: List[float],
    val_losses: List[float],
    train_dices: List[float],
    val_dices: List[float],
    save_path: str
):
    """
    Plots training and validation losses and Dice scores.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_dices: List of training Dice scores per epoch
        val_dices: List of validation Dice scores per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='s')
    plt.title("Loss (Lower is better)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Dice plot
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label="Train Dice", marker='o')
    plt.plot(val_dices, label="Val Dice", marker='s')
    plt.title("Dice Score (Higher is better)")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_predictions(predictions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], save_path: str):
    """
    Visualizes predictions and saves them to the specified path.
    
    Args:
        predictions: List of predictions (image, mask, prediction)
        save_path: Path to save the predictions (directory)
    """
    os.makedirs(save_path, exist_ok=True)
    for i, (image, mask, pred) in enumerate(predictions):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1); plt.imshow(image, cmap='gray'); plt.title("Image")
        plt.subplot(1, 3, 2); plt.imshow(mask, cmap='gray'); plt.title("Mask")
        plt.subplot(1, 3, 3); plt.imshow(pred, cmap='gray'); plt.title("Prediction")
        plt.savefig(os.path.join(save_path, f"prediction_{i+1}.png"))
        plt.close()
    
    # upload to wandb
    wandb.log({
        "predictions": [wandb.Image(os.path.join(save_path, f"prediction_{i+1}.png")) for i in range(len(predictions))]
    })
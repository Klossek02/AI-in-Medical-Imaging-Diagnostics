import torch

def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor):
    """
    Calculates Dice Score and Accuracy for binary segmentation.
    
    Args:
        preds: Predicted masks [B, H, W] or [B, 1, H, W] with class indices (0 or 1)
        targets: Ground truth masks [B, H, W] or [B, 1, H, W] with class indices (0 or 1)
    
    Returns:
        dice: Dice score (scalar tensor)
        acc: Accuracy (scalar tensor)
    """
    smooth = 1e-6
    
    # Remove channel dimension if present
    if targets.ndim == 4:
        targets = targets.squeeze(1)
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    
    # Flatten tensors
    preds_flat = preds.view(-1).float()
    targets_flat = targets.view(-1).float()
    
    # Dice Score
    intersection = (preds_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    
    # Accuracy
    correct = (preds_flat == targets_flat).sum()
    acc = correct / len(targets_flat)
    
    return dice, acc

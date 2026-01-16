import torch
from monai.metrics import DiceMetric, MeanIoU
from sklearn.metrics import accuracy_score
from src.config import Config

def calculate_metrics(config: Config, preds: torch.Tensor, targets: torch.Tensor):
    """
    Calculates Dice Score and Accuracy for multi-class segmentation.
    
    Args:
        config: Config object
        preds: Predicted masks [B, H, W] or [B, 1, H, W] with class indices (0, 1, 2, ...)
        targets: Ground truth masks [B, H, W] or [B, 1, H, W] with class indices (0, 1, 2, ...)
    
    Returns:
        mdice: Mean Dice score (scalar tensor) - mean across all classes
        acc: Accuracy (scalar tensor)
        miou: Mean IoU score (scalar tensor) - mean across all classes
        dice_per_class: Dice score per class (list of scalar tensors)
        iou_per_class: IoU score per class (list of scalar tensors)
    """
    # Remove channel dimension if present
    if targets.ndim == 4:
        targets = targets.squeeze(1)
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    
    # Convert class indices to one-hot encoding for MONAI metrics
    # MONAI expects one-hot format: [B, C, H, W] where C is num_classes
    num_classes = config.model.classes
    
    # Convert predictions to one-hot: [B, H, W] -> [B, C, H, W]
    preds_onehot = torch.nn.functional.one_hot(preds.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Convert targets to one-hot: [B, H, W] -> [B, C, H, W]
    targets_onehot = torch.nn.functional.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Initialize metrics with "mean" reduction to get single mean across all classes
    dice_metric = DiceMetric(reduction="mean", num_classes=num_classes)
    mean_iou = MeanIoU(reduction="mean")
    dice_per_class = DiceMetric(reduction="mean_batch", num_classes=num_classes)
    iou_per_class = MeanIoU(reduction="mean_batch")

    # Update metrics with batch data
    dice_metric(preds_onehot, targets_onehot)
    mean_iou(preds_onehot, targets_onehot)
    dice_per_class(preds_onehot, targets_onehot)
    iou_per_class(preds_onehot, targets_onehot)

    # Aggregate and get single mean values
    mdice = dice_metric.aggregate()
    miou = mean_iou.aggregate()
    dice_per_class = dice_per_class.aggregate()
    iou_per_class = iou_per_class.aggregate()
    
    # Calculate accuracy (doesn't need one-hot encoding)
    acc = accuracy_score(targets.cpu().view(-1).numpy(), preds.cpu().view(-1).numpy())
    
    return mdice, torch.tensor(acc, device=preds.device), miou, dice_per_class, iou_per_class

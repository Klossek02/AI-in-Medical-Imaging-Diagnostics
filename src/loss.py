from monai.losses import DiceFocalLoss, DiceLoss
import torch.nn as nn
from typing import Literal

def get_loss_fn(
    loss_fn_type: Literal["DiceFocalLoss", "DiceLoss", "CrossEntropyLoss"],
    **kwargs
):
    """
    Returns a loss function based on the specified type.
    
    Args:
        loss_fn_type: Type of loss function to use
        **kwargs: Additional parameters for the loss function
    
    Returns:
        Loss function instance
    """
    print(f"Using loss function: {loss_fn_type} with params {kwargs}")
    if loss_fn_type == "DiceFocalLoss":
        return DiceFocalLoss(**kwargs)
    elif loss_fn_type == "DiceLoss":
        return DiceLoss(**kwargs)
    elif loss_fn_type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function type: {loss_fn_type}")

import torch
import wandb
from src.config import Config
from src.models import build_model

def get_model(config: Config, device: torch.device = None):
    """
    Builds and returns a model based on the configuration.
    
    Args:
        config: Config object containing model configuration
        device: torch device (cuda or cpu). If None, uses config.training.device
    
    Returns:
        Model instance moved to the specified device
    """
    if device is None:
        device = torch.device(config.training.device)
    
    model = build_model(
        device=device,
        model_type=config.model.model_type,
        in_channels=config.model.in_channels,
        classes=config.model.classes,
        base_c=config.model.base_c,
        bilinear=config.model.bilinear
    )

    # print number of parameters
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    return torch.compile(model) if config.model.compile else model

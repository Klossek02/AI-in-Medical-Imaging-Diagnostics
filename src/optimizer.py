import torch.optim as optim
from src.config import Config

def get_optimizer(model, config: Config):
    """
    Creates an optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Config object
    
    Returns:
        Optimizer instance
    """
    print(f"Using optimizer: AdamW with learning rate {config.training.learning_rate} and weight decay {config.training.weight_decay}")
    return optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

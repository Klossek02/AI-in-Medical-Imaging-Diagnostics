import torch.optim as optim
from src.config import Config

def get_scheduler(optimizer, config: Config):
    """
    Creates a learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Config object
    
    Returns:
        Scheduler instance
    """
    print(f"Using scheduler: {config.training.scheduler_type} with params {config.training.scheduler_params}")
    if config.training.scheduler_type == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config.training.scheduler_params
        )
    elif config.training.scheduler_type == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            **{k: v for k, v in config.training.scheduler_params.items() if k != 'patience' and k != 'mode'}
        )
    elif config.training.scheduler_type == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.scheduler_params['step_size'],
            gamma=config.training.scheduler_params['gamma']
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.training.scheduler_type}")

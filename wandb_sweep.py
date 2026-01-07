"""
WandB Sweep CLI script for hyperparameter optimization.

Usage:
    # Initialize a sweep
    wandb sweep sweep_config.yaml

    # Run an agent
    wandb agent <sweep_id>
"""

import argparse
import os
import wandb
from src.config import Config
from src.preprocessing import preprocess_data
from src.dataloader import get_dataloaders
from src.train import train

SWEEPABLE_CONFIG_MAP: dict[str, str] = {
    'learning_rate': 'training.learning_rate',
    'weight_decay': 'training.weight_decay',
    'batch_size': 'dataloader.batch_size',
    'num_epochs': 'training.num_epochs',
    'scheduler_type': 'training.scheduler_type',
    'scheduler_patience': 'training.scheduler_params.patience',
    'scheduler_factor': 'training.scheduler_params.factor',
    'loss_fn_type': 'training.loss_fn_type',
    'model_type': 'model.model_type',
    'base_c': 'model.base_c',
    'early_stopping_patience': 'training.early_stopping_patience',
    'early_stopping_delta': 'training.early_stopping_delta',
    'distorsion_prob': 'dataloader.distorsion_prob',
    'distorsion_num_cells': 'dataloader.distorsion_num_cells',
    'distorsion_distort_limit': 'dataloader.distorsion_distort_limit',
    'noise_prob': 'dataloader.noise_prob',
    'noise_mean': 'dataloader.noise_mean',
    'noise_std': 'dataloader.noise_std',
    'contrast_prob': 'dataloader.contrast_prob',
    'contrast_gamma': 'dataloader.contrast_gamma',
    'rotation_prob': 'dataloader.rotation_prob',
    'flip_prob': 'dataloader.flip_prob',
    'crop_num_samples': 'dataloader.crop_num_samples',
    'crop_size': 'dataloader.crop_size',
    'crop_pos': 'dataloader.crop_pos',
    'crop_neg': 'dataloader.crop_neg',
}

def getattr_nested(obj, path, default=None):
    for attr in path.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            obj = obj[attr]
    return obj

def setattr_nested(obj, path, value):
    *parents, last = path.split(".")
    is_dict = False
    for attr in parents:
        try:
            obj = getattr(obj, attr)
            is_dict = False
        except AttributeError:
            is_dict = True
            obj = obj[attr]

    if not is_dict and isinstance(obj, dict):
        is_dict = True

    if is_dict:
        obj[last] = value
    else:
        setattr(obj, last, value)

def update_config_from_wandb(config: Config, wandb_config: dict):
    """
    Updates config values from wandb.config.
    Only updates values that are present in wandb.config.
    
    Args:
        config: Config object to update
        wandb_config: Dictionary from wandb.config
    """
    # Update training config
    for key, value in wandb_config.items():
        if value and key in SWEEPABLE_CONFIG_MAP:
            setattr_nested(config, SWEEPABLE_CONFIG_MAP[key], value)

def main():
    config = Config()
    parser = argparse.ArgumentParser(description="WandB Sweep Training Script")
    for key, value in SWEEPABLE_CONFIG_MAP.items():
        parser.add_argument(f"--{key}", type=type(getattr_nested(config, value)), default=None, help=f"{key} value")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = Config.from_json(f.read())

    update_config_from_wandb(config, args.__dict__)

    wandb_config = {}
    for key, value in SWEEPABLE_CONFIG_MAP.items():
        wandb_config[key] = getattr_nested(config, value)
    
    wandb.init(
        project=config.wandb.project,
        config=wandb_config
    )
    
    print(f"🔧 WandB Run: {wandb.run.name}")
    print(f"📊 WandB Config: {dict(wandb.config)}")
    
    # Preprocess data (only if needed, can be skipped if already preprocessed)
    processed_count = preprocess_data(config)
    if processed_count > 0:
        print(f"Preprocessed {processed_count} images")
    
    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # Train (wandb logging will be handled inside train function)
    save_dir = os.path.join(config.training.save_dir, wandb.run.name)
    os.makedirs(save_dir)
    train(config, train_loader, val_loader, test_loader, save_dir=save_dir, use_wandb=True)
    
    wandb.finish()


if __name__ == "__main__":
    main()

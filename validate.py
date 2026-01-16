import argparse
from monai.utils import set_determinism
from src.config import Config
from src.preprocessing import preprocess_data
from src.dataloader import get_dataloaders
from src.loss import get_loss_fn
from src.train import test

if __name__ == "__main__":
    # 1. Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = Config.from_json(f.read())
    
    preprocess_data(config)

    # get dataloaders
    _, _, test_loader = get_dataloaders(config)

    # Set random seed for reproducibility
    if config.training.seed is not None:
        set_determinism(seed=config.training.seed)
    
    save_dir = args.model_dir
    
    # Create loss function
    loss_fn = get_loss_fn(config.training.loss_fn_type, **config.training.loss_fn_params)
   
    # Test
    test(config, test_loader, loss_fn, config.training.device, vis_num=config.training.vis_num, save_dir=save_dir, use_wandb=False)

    print(f"\n✅ All done!")
    

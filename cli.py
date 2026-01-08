import argparse
import os
from src.config import Config
from src.preprocessing import preprocess_data
from src.dataloader import get_dataloaders
from src.train import train

if __name__ == "__main__":
    # 1. Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = Config.from_json(f.read())
    
    # 2. Preprocess data
    processed_count = preprocess_data(config)
    print(f"Preprocessed {processed_count} images")

    # 3. Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    # 4. Train
    save_dir = os.path.join("models", config.run_identifier)    
    os.makedirs(save_dir)
    train(config, train_loader, val_loader, test_loader, save_dir=save_dir, use_wandb=False)

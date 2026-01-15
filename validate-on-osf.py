import argparse
from monai.utils import set_determinism
from src.config import Config
from src.preprocessing import preprocess_osf
from src.dataloader import get_osf_dataloader
from src.loss import get_loss_fn
from src.train import test

def model_output_callback(outputs):
    print(outputs.shape)
    # its [B, C, H, W]
    outputs[:, 2, :, :] = outputs[:, 3, :, :]
    # set -inf to the channel with index 3
    outputs[:, 3, :, :] = float('-inf')
    return outputs

if __name__ == "__main__":
    # 1. Load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = Config.from_json(f.read())
    
    # overwrite preprocessed data directory so true datasets are not modified
    config.preprocessed_data_dir = "data/preprocessed-OSF-validation-files"

    # preprocess osf dataset
    preprocess_osf(config)

    # get dataloader
    dataloader = get_osf_dataloader(config)

    # Set random seed for reproducibility
    if config.training.seed is not None:
        set_determinism(seed=config.training.seed)
    
    save_dir = args.model_dir
    
    # Create loss function
    loss_fn = get_loss_fn(config.training.loss_fn_type, **config.training.loss_fn_params)
   
    # Test
    test(config, dataloader, loss_fn, config.training.device, vis_num=config.training.vis_num, save_dir=save_dir, use_wandb=False, model_output_callback=model_output_callback)

    print(f"\n✅ All done!")
    

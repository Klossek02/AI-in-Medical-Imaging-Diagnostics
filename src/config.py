import dataclasses
import dataclass_wizard
import torch
from typing import Optional, Literal
from src.models import models

ModelType = Literal[tuple(models.keys())]

@dataclasses.dataclass
class TrainingConfig(dataclass_wizard.JSONWizard):
    save_dir: str = "models"
    vis_num: int = 3 # number of examples to visualize every validation epoch
    early_stopping_patience: int = 3
    early_stopping_delta: float = 0.001
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    num_epochs: int = 15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = None
    scheduler_type: Literal["ReduceLROnPlateau", "CosineAnnealingLR", "StepLR"] = "ReduceLROnPlateau"
    scheduler_params: dict = dataclasses.field(default_factory=lambda: {"patience": 3, "factor": 0.5, "min_lr": 1e-6, "mode": "min"})
    loss_fn_type: Literal["DiceFocalLoss", "DiceLoss", "CrossEntropyLoss"] = "DiceFocalLoss"
    loss_fn_params: dict = dataclasses.field(default_factory=lambda: {"softmax": True, "to_onehot_y": True, "lambda_dice": 1.0, "lambda_focal": 2.0})

@dataclasses.dataclass
class PreprocessingConfig(dataclass_wizard.JSONWizard):
    spider_path: str = "data/SPIDER"
    spider_prefix: str = "SPIDER"
    spider_images_path: str = "data/SPIDER/images/images"
    spider_masks_path: str = "data/SPIDER/masks"
    osf_path: str = "data/osf-files"
    osf_prefix: str = "OSF"
    spine_output_path: str = "data/spine_output"
    spine_output_prefix: str = "Spine_Output"
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    min_pixels_mask: int = 50

@dataclasses.dataclass
class DataLoaderConfig(dataclass_wizard.JSONWizard):
    batch_size: int = 8
    num_workers: int = 2
    val_split: float = 0.2
    test_split: float = 0.1
    target_size: tuple[int, int] = (512, 512)
    distorsion_prob: float = 0.3
    distorsion_num_cells: int = 5
    distorsion_distort_limit: float = 0.03
    noise_prob: float = 0.1
    noise_mean: float = 0.0
    noise_std: float = 0.1
    contrast_prob: float = 0.2
    contrast_gamma: tuple[float, float] = (0.5, 2.0)
    rotation_prob: float = 0.5
    flip_prob: float = 0.5
    crop_size: tuple[int, int] = (256, 256)
    crop_pos: float = 1.0
    crop_neg: float = 1.0
    crop_num_samples: int = 4
    sw_overlap: float = 0.5

@dataclasses.dataclass
class ModelConfig(dataclass_wizard.JSONWizard):
    model_type: ModelType = "tri_conv_unext"
    in_channels: int = 1
    classes: int = 2
    base_c: int = 32
    bilinear: bool = True

@dataclasses.dataclass
class WandBConfig(dataclass_wizard.JSONWizard):
    project: str = "spine-segmentation"

@dataclasses.dataclass
class Config(dataclass_wizard.JSONWizard):
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = dataclasses.field(default_factory=PreprocessingConfig)
    dataloader: DataLoaderConfig = dataclasses.field(default_factory=DataLoaderConfig)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    wandb: WandBConfig = dataclasses.field(default_factory=WandBConfig)
    preprocessed_data_dir: str = "data/preprocessed_v2"
    cache_dir: str = "data/cache"
    run_identifier: str = "default"

# Generate default config when running python src/config.py
if __name__ == "__main__":
    config = Config()
    with open("default_config.json", "w") as f:
        f.write(config.to_json())
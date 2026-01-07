import argparse
import os
import glob
import numpy as np
from src.config import Config
from src.preprocessing import preprocess_data
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, PersistentDataset
from sklearn.model_selection import train_test_split

from monai.transforms import ( # reference: https://docs.wandb.ai/models/tutorials/monai_3d_segmentation; https://oss-ai-ml-medical-segmentation.readthedocs.io/en/latest/MONAI%20Tutorials/3D%20Segmentation%20-%20Spleen.html
    Compose,
    EnsureChannelFirstd,
    RandRotate90d,
    RandFlipd,
    EnsureTyped,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGridDistortiond, 
    ToTensord,
    Resized,
    LoadImaged,
    RandCropByPosNegLabeld)


def get_transforms(config: Config, mode="train"):
    """
    Function returning all MONAI transforms for training and validation & test.
    # reference: https://monai-dev.readthedocs.io/en/stable/transforms.html
    """

    if mode == "train":
        return Compose([ # TRAINING TRANSFORMS (AUGMENTATIONS)
            # load images and masks and adjust its types
            LoadImaged(keys=["image", "label"]),
            EnsureTyped(keys=["image"], dtype=np.float32),
            EnsureTyped(keys=["label"], dtype=np.uint8),
            # we make sure that we have [channels (C), height (H), width (W)]
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            
            # random crop by positive/negative label
            Resized(keys=["image", "label"], spatial_size=config.dataloader.target_size, mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(keys=["image", "label"],
                spatial_size=config.dataloader.crop_size,
                pos=config.dataloader.crop_pos,
                neg=config.dataloader.crop_neg,
                num_samples=config.dataloader.crop_num_samples,
                label_key="label"),
            # rotations, reflections
            RandRotate90d(keys=["image", "label"], prob=config.dataloader.rotation_prob, spatial_axes=[0, 1]),
            RandFlipd(keys=["image", "label"], prob=config.dataloader.flip_prob, spatial_axis=1),
            
            # mesh/grid distortions
            RandGridDistortiond(keys=["image", "label"],
                prob=config.dataloader.distorsion_prob,
                num_cells=config.dataloader.distorsion_num_cells,
                distort_limit=config.dataloader.distorsion_distort_limit,
                mode=("bilinear", "nearest")),
            
            # noise and contrast adjustments
            RandGaussianNoised(keys=["image"],
                prob=config.dataloader.noise_prob,
                mean=config.dataloader.noise_mean,
                std=config.dataloader.noise_std),
            RandAdjustContrastd(keys=["image"],
                prob=config.dataloader.contrast_prob,
                gamma=config.dataloader.contrast_gamma),
            ToTensord(keys=["image", "label"]),
        ])
    else: # VALIDATION AND TEST TRANSFORMS (NO AUGMENTATIONS!!!)
        return Compose([ # only resizing and tensor conversion
            LoadImaged(keys=["image", "label"]),
            EnsureTyped(keys=["image"], dtype=np.float32),
            EnsureTyped(keys=["label"], dtype=np.uint8),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            Resized(keys=["image", "label"], spatial_size=config.dataloader.target_size, mode=("bilinear", "nearest")),
            ToTensord(keys=["image", "label"]),
        ])


def get_dataloaders(config: Config):
    """
    Function to create dataloaders for training, validation and test sets. 
    """

    # we search recursively for .npy files in 'images' folders
    search_path = os.path.join(config.preprocessed_data_dir, "**", "images", "*.npy")
    all_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(all_files)} files (slices) in total.")
    
    if len(all_files) == 0:
    
        alt_path = os.path.join(config.preprocessed_data_dir, "*", "images", "*.npy")
        all_files = glob.glob(alt_path, recursive=True)

        if len(all_files) == 0:
            raise ValueError(f"There are no .npy files in: {config.preprocessed_data_dir}")


    # Splitting into train, val, test
    train_val_files, test_files = train_test_split(all_files, test_size=config.dataloader.test_split, random_state=33)
    train_files, val_files = train_test_split(train_val_files, test_size=config.dataloader.val_split, random_state=33)

    print(f"Split: training = {len(train_files)}, validation = {len(val_files)}, test = {len(test_files)}")

    # creating datasets
    # train_ds = SpineDataset(config, train_files, transform=get_transforms(config, "train"))
    # val_ds   = SpineDataset(config, val_files,   transform=get_transforms(config, "val"))
    # test_ds  = SpineDataset(config, test_files,  transform=get_transforms(config, "val"))

    # use cache dataset, make sure to load data as "label" and "image" like in SpineDataset
    def load_data(path):
        return {"image": path, "label": path.replace('images', 'masks')}
    train_files = [load_data(path) for path in train_files]
    val_files = [load_data(path) for path in val_files]
    test_files = [load_data(path) for path in test_files]
    
    train_ds = Dataset(data=train_files, transform=get_transforms(config, "train"))
    val_ds   = PersistentDataset(data=val_files, transform=get_transforms(config, "val"), cache_dir=config.cache_dir)
    test_ds  = PersistentDataset(data=test_files, transform=get_transforms(config, "val"), cache_dir=config.cache_dir)

    # creating dataloaders
    train_loader = DataLoader(train_ds, batch_size=config.dataloader.batch_size // config.dataloader.crop_num_samples, shuffle=True, num_workers=config.dataloader.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.dataloader.batch_size, shuffle=False, num_workers=config.dataloader.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.dataloader.batch_size, shuffle=False, num_workers=config.dataloader.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader

# TESTING
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = Config.from_json(f.read())

    # Preprocess data
    processed_count = preprocess_data(config)
    print(f"Preprocessed {processed_count} images")
    
    # Get dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(config)
    
    # Test dataloader
    data = next(iter(train_dl))
    print(len(data))

    images = torch.cat([d["image"] for d in data], dim=0)
    masks = torch.cat([d["label"] for d in data], dim=0)
    
    print(f"Dataloader test successful!")
    print(f" Image shape: {images.shape}")
    print(f" Shape masks: {masks.shape} ")
    print(f" Mask type: {masks.dtype}")
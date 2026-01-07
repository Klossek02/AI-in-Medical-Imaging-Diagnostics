import argparse
import os
import glob
import numpy as np
from src.config import Config
from src.preprocessing import preprocess_data
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from monai.transforms import ( # reference: https://docs.wandb.ai/models/tutorials/monai_3d_segmentation; https://oss-ai-ml-medical-segmentation.readthedocs.io/en/latest/MONAI%20Tutorials/3D%20Segmentation%20-%20Spleen.html
    Compose,
    EnsureChannelFirstd,
    RandRotate90d,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGridDistortiond, 
    ToTensord,
    Resized,
    RandCropByPosNegLabeld)

class SpineDataset(Dataset):
    def __init__(self, config: Config, file_paths, transform=None):
        self.config = config
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        mask_path = img_path.replace('images', 'masks')

        try:
            img_arr = np.load(img_path).astype(np.float32) # data normalized during preprocessing
            mask_arr = np.load(mask_path).astype(np.uint8)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(1, *self.config.dataloader.target_size), torch.zeros(*self.config.dataloader.target_size).long() # returning empty tensors on error


        # mask binarization
        mask_arr[mask_arr > 0] = 1

        data = {"image": img_arr, "label": mask_arr}

        if self.transform:
            data = self.transform(data)

        image = data["image"]
        mask = data["label"]
        
        mask = mask.long()

        return image, mask



def get_transforms(config: Config, mode="train"):
    """
    Function returning all MONAI transforms for training and validation & test.
    # reference: https://monai-dev.readthedocs.io/en/stable/transforms.html
    """

    if mode == "train":
        return Compose([ # TRAINING TRANSFORMS (AUGMENTATIONS)

            # we make sure that we have [channels (C), height (H), width (W)]
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            
            # random crop by positive/negative label
            RandCropByPosNegLabeld(keys=["image", "label"],
                spatial_size=config.dataloader.crop_size,
                pos=config.dataloader.crop_pos,
                neg=config.dataloader.crop_neg,
                num_samples=config.dataloader.crop_num_samples,
                label_key="label"),
            
            Resized(keys=["image", "label"], spatial_size=config.dataloader.target_size, mode=("bilinear", "nearest")),
            
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
    train_ds = SpineDataset(config, train_files, transform=get_transforms(config, "train"))
    val_ds   = SpineDataset(config, val_files,   transform=get_transforms(config, "val"))
    test_ds  = SpineDataset(config, test_files,  transform=get_transforms(config, "val"))

    # creating dataloaders
    train_loader = DataLoader(train_ds, batch_size=config.dataloader.batch_size, shuffle=True, num_workers=config.dataloader.num_workers, pin_memory=True)
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
    images, masks = next(iter(train_dl))
    
    print(f"Dataloader test successful!")
    print(f" Image shape: {images.shape}")
    print(f" Shape masks: {masks.shape} ")
    print(f" Mask type: {masks.dtype}")
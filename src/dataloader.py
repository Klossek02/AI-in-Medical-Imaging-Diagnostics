import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from src.config import Config
from src.preprocessing import preprocess_data
import torch
from torch.utils.data import DataLoader
from monai.data import Dataset, PersistentDataset, CacheDataset
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
    ResizeWithPadOrCropd,
    LoadImaged,
    RandCropByPosNegLabeld)


def collate_fn(batch):
    """
    Custom collate function as a safety net for dtype consistency.
    ResizeWithPadOrCropd in transforms should ensure all tensors have the same size,
    but we ensure dtype consistency here as well.
    """
    images = []
    labels = []
    
    for item in batch:
        if isinstance(item, dict):
            img = item["image"]
            lbl = item["label"]
        else:
            img = torch.cat([d["image"] for d in item], dim=0)
            lbl = torch.cat([d["label"] for d in item], dim=0)
        
        # Ensure consistent dtype (safety net - transforms should handle this)
        if img.dtype != torch.float32:
            img = img.float()
        if lbl.dtype != torch.long:
            # Handle uint8 -> long conversion
            lbl = lbl.long()
        
        images.append(img)
        labels.append(lbl)
    
    # Stack into batch tensors (should work since ResizeWithPadOrCropd ensures same sizes)
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return {"image": images, "label": labels}


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
            RandCropByPosNegLabeld(keys=["image", "label"],
                spatial_size=config.dataloader.crop_size,
                pos=config.dataloader.crop_pos,
                neg=config.dataloader.crop_neg,
                num_samples=config.dataloader.crop_num_samples,
                allow_smaller=True,
                label_key="label"),
            # ResizeWithPadOrCropd ensures exact size (better than SpatialPadd which only pads to minimum)
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config.dataloader.crop_size, mode=("constant", "constant")),
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
            # Convert to tensors (dtype conversion to torch.long for labels happens in collate_fn)
            ToTensord(keys=["image", "label"]),
        ])
    else: # VALIDATION AND TEST TRANSFORMS (NO AUGMENTATIONS!!!) 
        return Compose([ # only resizing and tensor conversion
            LoadImaged(keys=["image", "label"]),
            EnsureTyped(keys=["image"], dtype=np.float32),
            EnsureTyped(keys=["label"], dtype=np.uint8),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            # ResizeWithPadOrCropd ensures exact size (better than SpatialPadd which only pads to minimum)
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=config.dataloader.crop_size, mode=("constant", "constant")),
            # Convert to tensors (dtype conversion to torch.long for labels happens in collate_fn)
            ToTensord(keys=["image", "label"]),
        ])
def load_data(path):
    return {"image": path, "label": path.replace('images', 'masks')}

def get_osf_dataloader(config: Config):
    """
    Function to create dataloader for OSF dataset.
    """
    osf_files = [load_data(path) for path in glob.glob(os.path.join(config.preprocessed_data_dir, "**", "images", "*.npy"), recursive=True)]
    osf_ds = Dataset(data=osf_files, transform=get_transforms(config, "val"))
    return DataLoader(osf_ds, batch_size=config.dataloader.batch_size, shuffle=False, num_workers=config.dataloader.num_workers, pin_memory=True, collate_fn=collate_fn)

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
    
    # split files by patient
    patient_files = {}
    for file in all_files:
        patient_id = os.path.basename(file).split("_slice")[0]
        if patient_id not in patient_files:
            patient_files[patient_id] = []
        patient_files[patient_id].append(file)
    
    patient_ids = list(patient_files.keys())

    train_val_patient_ids, test_patient_ids = train_test_split(patient_ids, test_size=config.dataloader.test_split, random_state=33)
    train_patient_ids, val_patient_ids = train_test_split(train_val_patient_ids, test_size=config.dataloader.val_split, random_state=33)

    print(f"Split (num of patients): training = {len(train_patient_ids)}, validation = {len(val_patient_ids)}, test = {len(test_patient_ids)}")

    train_files = [load_data(file) for pid in train_patient_ids for file in patient_files[pid]]
    val_files = [load_data(file) for pid in val_patient_ids for file in patient_files[pid]]
    test_files = [load_data(file) for pid in test_patient_ids for file in patient_files[pid]]

    print(f"Split (num of files): training = {len(train_files)}, validation = {len(val_files)}, test = {len(test_files)}")
    
    train_ds = Dataset(data=train_files, transform=get_transforms(config, "train"))
    val_ds   = CacheDataset(data=val_files, transform=get_transforms(config, "val"), cache_rate=1.0, num_workers=config.dataloader.num_workers)
    test_ds  = PersistentDataset(data=test_files, transform=get_transforms(config, "val"), cache_dir=config.cache_dir)

    # creating dataloaders with custom collate function
    train_loader = DataLoader(train_ds, batch_size=config.dataloader.batch_size // config.dataloader.crop_num_samples, shuffle=True, num_workers=config.dataloader.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=config.dataloader.batch_size, shuffle=False, num_workers=config.dataloader.num_workers, pin_memory=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=config.dataloader.batch_size, shuffle=False, num_workers=config.dataloader.num_workers, pin_memory=True, collate_fn=collate_fn)

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
    try:
        data = next(iter(train_dl))
        images = data["image"]
        masks = data["label"]
        
        print(f"Dataloader test successful!")
        print(f" Image shape: {images.shape}")
        print(f" Image dtype: {images.dtype}")
        print(f" Mask shape: {masks.shape}")
        print(f" Mask dtype: {masks.dtype}")
        print(f" Image value range: [{images.min().item():.4f}, {images.max().item():.4f}]")
        print(f" Mask value range: [{masks.min().item()}, {masks.max().item()}]")
    except Exception as e:
        print(f"Dataloader test failed with error: {e}")
        raise
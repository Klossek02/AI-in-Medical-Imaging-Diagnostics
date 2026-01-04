import os
import glob
import numpy as np
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
    Resized)


TARGET_SIZE = (256, 256)

class SpineDataset(Dataset):

    def __init__(self, file_paths, transform=None):
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
            return torch.zeros(1, *TARGET_SIZE), torch.zeros(*TARGET_SIZE).long() # returning empty tensors on error


        # mask binarization
        mask_arr[mask_arr > 0] = 1

        data = {"image": img_arr, "label": mask_arr}

        if self.transform:
            data = self.transform(data)

        image = data["image"]
        mask = data["label"]
        
        mask = mask.long()

        return image, mask



def get_transforms(mode="train"):
    """
    Function returning all MONAI transforms for training and validation & test.
    # reference: https://monai-dev.readthedocs.io/en/stable/transforms.html
    """

    if mode == "train":
        return Compose([ # TRAINING TRANSFORMS (AUGMENTATIONS)

            # we make sure that we have [channels (C), height (H), width (W)]
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            Resized(keys=["image", "label"], spatial_size=TARGET_SIZE, mode=("bilinear", "nearest")),
            
            # rotations, reflections
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            
            # mesh/grid distortions
            RandGridDistortiond(keys=["image", "label"], prob=0.3, num_cells=5, distort_limit=0.03, mode=("bilinear", "nearest")),
            
            # noise and contrast adjustments
            RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 2.0)),
            
            ToTensord(keys=["image", "label"]),
        ])
    else: # VALIDATION AND TEST TRANSFORMS (NO AUGMENTATIONS!!!)
        
        return Compose([ # only resizing and tensor conversion
            
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            Resized(keys=["image", "label"], spatial_size=TARGET_SIZE, mode=("bilinear", "nearest")),
            ToTensord(keys=["image", "label"]),
        ])



def get_dataloaders(data_dir, batch_size=16, val_split=0.2, test_split=0.1, num_workers=2):
    """
    Function to create dataloaders for training, validation and test sets. 
    """

    # we search recursively for .npy files in 'images' folders
    search_path = os.path.join(data_dir, "**", "images", "*.npy")
    all_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(all_files)} files (slices) in total.")
    
    if len(all_files) == 0:
    
        alt_path = os.path.join(data_dir, "*", "images", "*.npy")
        all_files = glob.glob(alt_path, recursive=True)

        if len(all_files) == 0:
            raise ValueError(f"There are no .npy files in: {data_dir}")


    # Splitting into train, val, test
    train_val_files, test_files = train_test_split(all_files, test_size=test_split, random_state=33)
    train_files, val_files = train_test_split(train_val_files, test_size=val_split, random_state=33)

    print(f"Split: training = {len(train_files)}, validation = {len(val_files)}, test = {len(test_files)}")

    # creating datasets
    train_ds = SpineDataset(train_files, transform=get_transforms("train"))
    val_ds   = SpineDataset(val_files,   transform=get_transforms("val"))
    test_ds  = SpineDataset(test_files,  transform=get_transforms("val"))

    # creating dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    return train_loader, val_loader, test_loader



if __name__ == "__main__":
 
    TEST_DATA_DIR = "data/preprocessed_v2"  # REMEBER TO ADJUST TO YOURS!!
    
    if os.path.exists(TEST_DATA_DIR):
        try:
            train_dl, val_dl, test_dl = get_dataloaders(TEST_DATA_DIR, batch_size=4, num_workers=0)
            
            images, masks = next(iter(train_dl))
            
            print(f"Dataloader test successful!")
            print(f" Image shape: {images.shape}") # exp: [B, 1, 256, 256]
            print(f" Shape masks: {masks.shape} ") # exp: [B, 1, 256, 256]
            print(f" Mask type: {masks.dtype}")  # exp: torch.int64 / Long
            
        except Exception as e:
            print(f"Error during test: {e}")
            
    else:
        print(f"The folder cannot be found at {TEST_DATA_DIR}. The test is skipped.")
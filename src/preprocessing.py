import argparse
import numpy as np
import SimpleITK as sitk
import os
import glob
import shutil
from tqdm import tqdm

from src.config import Config

def preprocess_data(config: Config) -> int:
    processed_count = 0
    processed_count += preprocess_spider(config)
    # we decided to not use osf dataset, since it do not have spinal canal and C6/C7 labels
    # processed_count += preprocess_osf(config)
    processed_count += preprocess_spine_output(config)
    print(f"Processed {processed_count} images")
    return processed_count

def preprocess_spider(config: Config) -> int:
    if not print_info_and_early_return(config, config.preprocessing.spider_prefix, config.preprocessing.spider_path):
        return 0
    
    image_files = glob.glob(os.path.join(config.preprocessing.spider_images_path, "*.mha"))

    processed_count = 0

    for img_path in tqdm(image_files, desc=f"Processing {config.preprocessing.spider_prefix}"):

        filename = os.path.basename(img_path)

        if "t2" not in filename.lower(): continue 
        mask_path = os.path.join(config.preprocessing.spider_masks_path, filename)

        if os.path.exists(mask_path):
            try:
                img = sitk.ReadImage(img_path)
                mask = sitk.ReadImage(mask_path)
                # change mask to 1 for [1,25] and [101,125], to 2 for [100,100] and to 3 for [201,225] in sitk
                # fully visible vertebrae
                mask_1 = sitk.BinaryThreshold(mask, 1, 25, 1, 0)
                # partially visible vertebrae
                # mask_2 = sitk.BinaryThreshold(mask, 101, 125, 1, 0)
                # spinal canal
                mask_3 = sitk.BinaryThreshold(mask, 100, 100, 2, 0)
                # intervertebral discs
                mask_4 = sitk.BinaryThreshold(mask, 201, 225, 3, 0)
                mask = sitk.Add(mask_1, mask_3) # 1 or 2
                # mask = sitk.Add(mask, mask_3) # 1 or 2
                mask = sitk.Add(mask, mask_4) # 1, 2 or 3
                fname_clean = filename.replace('.mha', '')

                process_and_save(config, img, mask, f"{config.preprocessing.spider_prefix}_{fname_clean}", config.preprocessing.spider_prefix)

                processed_count += 1
            except Exception as e: print(f"Error {filename}: {e}")

    print(f"Processed {processed_count} {config.preprocessing.spider_prefix} images")
    return processed_count


def preprocess_osf(config: Config) -> int:
    if not print_info_and_early_return(config, config.preprocessing.osf_prefix, config.preprocessing.osf_path):
        return 0

    t2_candidates = []

    for root, dirs, files in os.walk(config.preprocessing.osf_path):
        for file in files:
            lower = file.lower()

            if "t2" in lower and (lower.endswith(".nii") or lower.endswith(".nii.gz")) and "fat" not in lower and "water" not in lower and "label" not in lower and "seg" not in lower:
                t2_candidates.append(os.path.join(root, file))
    
    processed_count = 0

    for t2_image_path in tqdm(t2_candidates, desc=config.preprocessing.osf_prefix):
        try:
            t2_filename = os.path.basename(t2_image_path)
            root = os.path.dirname(t2_image_path)
            parent = os.path.dirname(root) 
            labels_dir = os.path.join(parent, "Labels")

            if not os.path.exists(labels_dir): labels_dir = os.path.join(parent, "Masks")
            
            if os.path.exists(labels_dir):
                mask_files = glob.glob(os.path.join(labels_dir, "*.nii*"))
                if not mask_files: continue
                
                img = sitk.ReadImage(t2_image_path)
                combined_mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
                combined_mask.CopyInformation(img)
                
                masks_merged = 0


                for mf in mask_files:

                    try:
                        part_mask = sitk.ReadImage(mf)
                        part_mask = sitk.Cast(part_mask, sitk.sitkUInt8)
                        part_mask.SetOrigin(combined_mask.GetOrigin())
                        part_mask.SetSpacing(combined_mask.GetSpacing())
                        part_mask.SetDirection(combined_mask.GetDirection())
                        combined_mask = sitk.Add(combined_mask, part_mask)
                        masks_merged += 1
                    except: pass


                if masks_merged > 0:
                    combined_mask = sitk.BinaryThreshold(combined_mask, 1, 255, 1, 0)
                    folder_id = os.path.basename(os.path.dirname(parent))
                    scanner = os.path.basename(parent)
                    fname = t2_filename.replace('.nii.gz', '').replace('.nii', '')

                    process_and_save(config, img, combined_mask, f"{config.preprocessing.osf_prefix}_{folder_id}_{scanner}_{fname}", config.preprocessing.osf_prefix)

                    processed_count += 1
        except Exception as e: print(f"Error {t2_image_path}: {e}")
    print(f"Processed {processed_count} {config.preprocessing.osf_prefix} images")
    return processed_count

def preprocess_spine_output(config: Config) -> int:
    if not print_info_and_early_return(config, config.preprocessing.spine_output_prefix, config.preprocessing.spine_output_path):
        return 0
    
    processed_count = 0
    patient_dirs = glob.glob(os.path.join(config.preprocessing.spine_output_path, "SPINE_*"))
    for p_dir in tqdm(patient_dirs, desc=f"Processing {config.preprocessing.spine_output_prefix}"):
        try:
            patient_id = os.path.basename(p_dir) 
            img_path = os.path.join(p_dir, f"{patient_id}_sagittal_image.nii.gz")
            mask_path = os.path.join(p_dir, f"{patient_id}_sagittal_label.nii.gz")

            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = sitk.ReadImage(img_path)
                mask = sitk.ReadImage(mask_path)

                if process_and_save(config, img, mask, f"{config.preprocessing.spine_output_prefix}_{patient_id}", config.preprocessing.spine_output_prefix) > 0:
                    processed_count += 1

        except Exception as e: print(f"Error {patient_id}: {e}")
    print(f"Processed {processed_count} {config.preprocessing.spine_output_prefix} images")
    return processed_count


def resample_volume(image: sitk.Image, target_spacing: tuple = (1.0, 1.0, 1.0), is_label: bool = False) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [int(round(osz * osp / tsp)) for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)]

    resampler = sitk.ResampleImageFilter()  # reference: https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1ResampleImageFilter.html
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear) 

    return resampler.Execute(image)


def normalize_image(img_arr):
    """
    Min-max normalization (0-1 scailing).
    Reference: https://www.codecademy.com/article/min-max-zscore-normalization
    """
    
    # we remove negative values first
    img_arr[img_arr < 0] = 0

    # we remove outliers - the most bright pixels + taking 99 percentile from entire img
    p99 = np.percentile(img_arr, 99.0)
    if p99 > 0:
        img_arr = np.clip(img_arr, 0, p99)
    
    # mix-max scailing to [0, 1]
    min_val = np.min(img_arr)
    max_val = np.max(img_arr)
    
    if max_val - min_val > 0:
        img_arr = (img_arr - min_val) / (max_val - min_val)
    else:
        img_arr[:] = 0 # if img is empty
        
    return img_arr.astype(np.float32)


def process_and_save(config: Config, img_sitk: sitk.Image, mask_sitk: sitk.Image, filename: str, subset_name: str) -> int:
    # 1. resampling
    img_sitk = resample_volume(img_sitk, is_label=False)
    mask_sitk = resample_volume(mask_sitk, is_label=True)

    # 2. conversion to numpy
    arr_img = sitk.GetArrayFromImage(img_sitk)
    arr_mask = sitk.GetArrayFromImage(mask_sitk)

    # 3. detecting slice axis
    slice_axis = np.argmin(arr_img.shape)
    
    # 4. normalization 
    arr_img = normalize_image(arr_img)

    # 5. saving
    saved_count = 0
    num_slices = arr_img.shape[slice_axis]
    

    out_img_dir = os.path.join(config.preprocessed_data_dir, subset_name, "images")
    out_mask_dir = os.path.join(config.preprocessed_data_dir, subset_name, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)


    for i in range(num_slices):
        if slice_axis == 0:
            sl_img = arr_img[i, :, :]
            sl_mask = arr_mask[i, :, :]
        elif slice_axis == 1:
            sl_img = arr_img[:, i, :]
            sl_mask = arr_mask[:, i, :]
        else:
            sl_img = arr_img[:, :, i]
            sl_mask = arr_mask[:, :, i]


        if np.sum(sl_mask) > config.preprocessing.min_pixels_mask:
            fname_out = f"{filename}_slice{i:03d}.npy"
            np.save(os.path.join(out_img_dir, fname_out), sl_img.astype(np.float32))
            np.save(os.path.join(out_mask_dir, fname_out), sl_mask.astype(np.uint8))
            saved_count += 1
            
    return saved_count

def print_info_and_early_return(config: Config, prefix: str, data_dir: str) -> bool:
    if not os.path.exists(data_dir):
        print(f"{prefix} path {data_dir} does not exist")
        return False
    
    if os.path.exists(os.path.join(config.preprocessed_data_dir, prefix)):
        print(f"{prefix} data already preprocessed")
        return False
    
    print(f"\nPREPROCESSING {prefix} DATASET")
    return True

# TESTING
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = Config.from_json(f.read())

    # Preprocess data
    print(f"Preprocessing data...")
    processed_count = preprocess_data(config)
    print(f"Processed {processed_count} images")
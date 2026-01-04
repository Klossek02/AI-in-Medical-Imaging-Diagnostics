import numpy as np
import SimpleITK as sitk
import os
import glob
import shutil
from tqdm import tqdm


PATH_SPIDER = "data/SPIDER"
PATH_SPIDER_IMAGES = "data/SPIDER/images/images"
PATH_SPIDER_MASKS = "data/SPIDER/masks"

PATH_OSF = "data/osf-files" 
PATH_SPINE_OUTPUT = "data/spine_output"

OUTPUT_DIR = "data/preprocessed_v2"

TARGET_SPACING = (1.0, 1.0, 1.0) 
MIN_PIXELS_MASK = 50 



def resample_volume(image, target_spacing=TARGET_SPACING, is_label=False):

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



def process_and_save(img_sitk, mask_sitk, filename, subset_name):

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
    

    out_img_dir = os.path.join(OUTPUT_DIR, subset_name, "images")
    out_mask_dir = os.path.join(OUTPUT_DIR, subset_name, "masks")
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


        if np.sum(sl_mask) > MIN_PIXELS_MASK:
            fname_out = f"{filename}_slice{i:03d}.npy"
            np.save(os.path.join(out_img_dir, fname_out), sl_img.astype(np.float32))
            np.save(os.path.join(out_mask_dir, fname_out), sl_mask.astype(np.uint8))
            saved_count += 1
            
    return saved_count




def preprocess_spider():

    print("\nPREPROCESSING SPIDER DATASET")

    image_files = glob.glob(os.path.join(PATH_SPIDER_IMAGES, "*.mha"))

    processed_count = 0

    for img_path in tqdm(image_files, desc="Processing SPIDER"):

        filename = os.path.basename(img_path)

        if "t2" not in filename.lower(): continue 
        mask_path = os.path.join(PATH_SPIDER_MASKS, filename)

        if os.path.exists(mask_path):
            try:
                img = sitk.ReadImage(img_path)
                mask = sitk.ReadImage(mask_path)
                fname_clean = filename.replace('.mha', '')

                process_and_save(img, mask, f"Spider_{fname_clean}", "SPIDER")

                processed_count += 1
            except Exception as e: print(f"Error {filename}: {e}")



def preprocess_osf():

    print("\nPREPROCESSING OSF DATASET")

    t2_candidates = []

    for root, dirs, files in os.walk(PATH_OSF):
        for file in files:
            lower = file.lower()

            if "t2" in lower and (lower.endswith(".nii") or lower.endswith(".nii.gz")) and "fat" not in lower and "water" not in lower and "label" not in lower and "seg" not in lower:
                t2_candidates.append(os.path.join(root, file))
    
    processed_count = 0


    for t2_image_path in tqdm(t2_candidates, desc="OSF"):
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

                    process_and_save(img, combined_mask, f"OSF_{folder_id}_{scanner}_{fname}", "OSF")

                    processed_count += 1
        except Exception as e: print(f"Error {t2_image_path}: {e}")




def preprocess_spine_output():

    print("\n PREPROCESSING SPINE_OUTPUT DATASET")

    processed_count = 0


    patient_dirs = glob.glob(os.path.join(PATH_SPINE_OUTPUT, "SPINE_*"))

    for p_dir in tqdm(patient_dirs, desc="Processing Spine_Output"):

        try:
            patient_id = os.path.basename(p_dir) 
            img_path = os.path.join(p_dir, f"{patient_id}_sagittal_image.nii.gz")
            mask_path = os.path.join(p_dir, f"{patient_id}_sagittal_label.nii.gz")

            if os.path.exists(img_path) and os.path.exists(mask_path):
                img = sitk.ReadImage(img_path)
                mask = sitk.ReadImage(mask_path)

                if process_and_save(img, mask, f"Spine_output_{patient_id}", "Spine_Output") > 0:

                    processed_count += 1

        except Exception as e: print(f"Error {patient_id}: {e}")




if __name__ == "__main__":
    
    preprocess_spider()
    preprocess_osf()
    preprocess_spine_output()
    
    print(f"Data saved in: {OUTPUT_DIR}")
import numpy as np
import SimpleITK as sitk
import os
import glob
import shutil
from tqdm import tqdm


PATH_SPIDER = "data/SPIDER"
PATH_SPIDER_IMAGES = "data/SPIDER/images/images"
PATH_SPIDER_MASKS = "data/SPIDER/masks/masks"

PATH_OSF = "data/osf-files" 
PATH_SPINE_OUTPUT = "data/spine_output/spine_output"

OUTPUT_DIR = "data/preprocessed"

TARGET_SPACING = (1.0, 1.0, 1.0) # resampling to 1 mm isotropic
MIN_PIXELS_MASK = 50  # min num of pixels in mask to save slice 


def resample_volume(image, target_spacing=TARGET_SPACING, is_label=False):
    """
    Changing (resampling) the resolution of the image to 1 x 1 x 1 mm.
    Pictures: linear interpolation
    Masks: nearest neighbor interpolation to preserve classes 0,1,2
    """

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * osp / tsp))
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)]


    resampler = sitk.ResampleImageFilter()
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
    Z-score normalization: (x - mean) / std.
    Ignoring background (assumed to be values <= 10)
    """

    mask_brain = img_arr > 10 # we assume that the background is  < 10
    
    if np.sum(mask_brain) == 0:
        return img_arr
        
    mean = np.mean(img_arr[mask_brain])
    std = np.std(img_arr[mask_brain])
    
    if std == 0: return img_arr
    return (img_arr - mean) / std


def process_and_save(img_sitk, mask_sitk, filename, subset_name):
    """
    MAIN LOGIC: resampling -> normalization -> 3D to 2D slicing -> saving.
    """

    # 1. resampling
    img_sitk = resample_volume(img_sitk, is_label=False)
    mask_sitk = resample_volume(mask_sitk, is_label=True)

    # 2. conversion to Numpy (SimpleITK to Z, Y, X)
    arr_img = sitk.GetArrayFromImage(img_sitk)
    arr_mask = sitk.GetArrayFromImage(mask_sitk)

    # 3. Detecting slice axis (Robustness!)
    # In EDA we saw that slices are the smallest dimension (like 15-25)
    slice_axis = np.argmin(arr_img.shape)
    
    # 4. normalization
    arr_img = normalize_image(arr_img)

    # 5. iteration, saving non-empty slices
    saved_count = 0
    num_slices = arr_img.shape[slice_axis]
    

    out_img_dir = os.path.join(OUTPUT_DIR, subset_name, "images")
    out_mask_dir = os.path.join(OUTPUT_DIR, subset_name, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)


    for i in range(num_slices):
        # taking 2D slice based on detected axis

        if slice_axis == 0:
            sl_img = arr_img[i, :, :]
            sl_mask = arr_mask[i, :, :]
        elif slice_axis == 1:
            sl_img = arr_img[:, i, :]
            sl_mask = arr_mask[:, i, :]
        else:
            sl_img = arr_img[:, :, i]
            sl_mask = arr_mask[:, :, i]


        # 6. filtering empty masks 
        # we save only if the spine is visible in the mask
        if np.sum(sl_mask) > MIN_PIXELS_MASK:

            fname_out = f"{filename}_slice{i:03d}.npy"
            np.save(os.path.join(out_img_dir, fname_out), sl_img.astype(np.float32))
            np.save(os.path.join(out_mask_dir, fname_out), sl_mask.astype(np.uint8))
            saved_count += 1
            
    return saved_count


###########################################
# SPIDER PREPROCESSING
############################################
def preprocess_spider():

    print("\nPREPROCESSING SPIDER DATASET")
    
    # we download all .mha files from images and masks 
    image_files = glob.glob(os.path.join(PATH_SPIDER_IMAGES, "*.mha"))
    
    print(f"Found {len(image_files)} files in the images folder (T1 + T2).")
    
    processed_count = 0

    for img_path in tqdm(image_files, desc="Processing SPIDER"):
        filename = os.path.basename(img_path)
        
        # we filter only T2 files
        if "t2" not in filename.lower():
            continue 

        mask_path = os.path.join(PATH_SPIDER_MASKS, filename)
        
        if os.path.exists(mask_path):
            try:
                img = sitk.ReadImage(img_path)
                mask = sitk.ReadImage(mask_path)
                
                fname_clean = filename.replace('.mha', '')
                unique_name = f"Spider_{fname_clean}"
                
                process_and_save(img, mask, unique_name, "SPIDER")
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        else:
            # print(f"Missing mask for: {filename}")
            pass

    print(f"Processed {processed_count} T2 files of image-mask pairs.")


############################################
# OSF PREPROCESSING
############################################
def preprocess_osf():

    print("\nPREPROCESSING OSF DATASET")
    
    t2_candidates = []

    for root, dirs, files in os.walk(PATH_OSF):
        for file in files:
            lower = file.lower()
            if "t2" in lower and (lower.endswith(".nii") or lower.endswith(".nii.gz")) and "fat" not in lower and "water" not in lower and "label" not in lower and "seg" not in lower:
                t2_candidates.append(os.path.join(root, file))
    
    print(f"Found {len(t2_candidates)} potential T2 volumes.")


    processed_count = 0
    
    for t2_image_path in tqdm(t2_candidates, desc="OSF"):
        t2_filename = os.path.basename(t2_image_path)
        root = os.path.dirname(t2_image_path)
        
        parent = os.path.dirname(root) 
        labels_dir = os.path.join(parent, "Labels")
        if not os.path.exists(labels_dir): labels_dir = os.path.join(parent, "Masks")
            
        if os.path.exists(labels_dir):
            try:
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
                    except Exception as e:
                        # print(f"Skip part {os.path.basename(mf)}: {e}")
                        pass

                if masks_merged > 0:
                    combined_mask = sitk.BinaryThreshold(combined_mask, 1, 255, 1, 0)
                    
                    folder_id = os.path.basename(os.path.dirname(parent))
                    scanner = os.path.basename(parent)
                    fname = t2_filename.replace('.nii.gz', '').replace('.nii', '')
                    
                    name = f"OSF_{folder_id}_{scanner}_{fname}"
                    
                    process_and_save(img, combined_mask, name, "OSF")
                    processed_count += 1

            except Exception as e: 
                print(f"Error {t2_filename}: {e}")
                
    print(f"Processed {processed_count} OSF volumes.")


############################################
# SPINE_OUTPUT PROCESSING 
############################################
def preprocess_spine_output():

    print("\n PREPROCESSING SPINE_OUTPUT DATASET")
    
    processed_count = 0
    
    patient_dirs = glob.glob(os.path.join(PATH_SPINE_OUTPUT, "SPINE_*"))
    
    print(f"Found {len(patient_dirs)} patients' folders.")

    for p_dir in tqdm(patient_dirs, desc="Processing NewDataset"):
        patient_id = os.path.basename(p_dir) 
        
        img_name = f"{patient_id}_sagittal_image.nii.gz"
        mask_name = f"{patient_id}_sagittal_label.nii.gz"
        
        img_path = os.path.join(p_dir, img_name)
        mask_path = os.path.join(p_dir, mask_name)
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            try:
                img = sitk.ReadImage(img_path)
                mask = sitk.ReadImage(mask_path)
                
                fname = f"Spine_output_{patient_id}"
                
                saved_count = process_and_save(img, mask, fname, "Spine_Output")
                
                if saved_count > 0:
                    processed_count += 1
                else:
                    # print(f"0 slices saved for {fname} (empty mask).")
                    pass
                    
            except Exception as e:
                print(f"ERROR in {patient_id}: {e}")
        else:
            print(f"Missing files in {p_dir}")

    print(f"Processed {processed_count} T2 files of image-mask pairs.")



if __name__ == "__main__":
   
    # preprocess_spider()
    preprocess_osf()
    preprocess_spine_output()

    print(f"Data saved in: {OUTPUT_DIR}")
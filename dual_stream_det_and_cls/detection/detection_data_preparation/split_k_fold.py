
"""
This script prepares the dataset for 5-fold cross-validation.

1.  `create_k_folds_by_patient`: Splits all available images into 5 folds,
    ensuring that all images from a single patient belong to the same fold.
    This prevents data leakage between training and validation sets. It creates
    `fold_0.txt` through `fold_4.txt` in the `dataset_org` directory.

2.  `generate_per_fold_annotations`: For each fold `k`, this function creates two
    annotation files in the `dataset` directory:
    - `val_annotations_fold_k.txt`: Contains only the `_resized` images
      from the k-th fold, to be used for validation.
    - `train_annotations_fold_k.txt`: Contains all augmented images (excluding
      `_resized`) from the other four folds, to be used for training.
"""
import os

import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from ...immutables import ProjectPaths
from .detection_dataset_definitions import mass_union_mass_enhancement

AUGMENTATION_SUFFIXES = [
    "_hflip_noise",
    "_hvflip_brightcont",
    '_resized',
    "_symm_shit_blur",
    '_rotate20_elastic',
    "_gridshuffle",
    '_cutout',
    # '_cutout_mulnoise',
    # '_rotate20',
    # '_hflip',
    # '_vflip',
    # '_hvflip', 
    # '_blur',
    # '_brightcont',
    # '_shift',
    # "_hflip_cutout",
    # "_shift_blur",
]


def parse_image_name(image_name_from_file):
    """
    Parses a full image name to extract its base name and augmentation suffix.
    The "base name" includes the patient, side, view, and CM/DM marker.

    Args:
        image_name_from_file (str): The full image name, e.g., "P1_L_DM_MLO_resized.jpg".

    Returns:
        tuple: A tuple containing (base_name, augmentation_suffix).
               - base_name (str): e.g., "P1_L_DM_MLO"
               - augmentation_suffix (str): e.g., "_resized" or "" if no suffix.
    """
    name_no_ext = image_name_from_file.split('.')[0]
    
    # Check for augmentation suffixes
    for suffix in AUGMENTATION_SUFFIXES:
        if name_no_ext.endswith(suffix):
            base_name = name_no_ext[:-len(suffix)]
            return base_name, suffix
            
    # If no suffix is found, it's an original image
    return name_no_ext, ""


def create_k_folds_by_patient(k=5):
    """
    Splits the dataset into k folds based on Patient_ID to prevent data leakage.
    Saves the image names for each fold into separate text files.
    """
    print(f"--- Creating {k}-Fold Split by Patient ID ---")
    annotations = pd.read_csv(ProjectPaths.annotations_all_sheet_modified)
    annotations = annotations[annotations['Image_name'].isin(mass_union_mass_enhancement)]

    patient_groups = annotations.groupby('Patient_ID')['Image_name'].apply(list)
    patient_ids = patient_groups.index.to_numpy()
    
    gkf = GroupKFold(n_splits=k)
    
    print(f"Splitting {len(patient_ids)} patients into {k} folds...")
    for i, (_, val_indices) in enumerate(tqdm(gkf.split(patient_ids, groups=patient_ids), total=k, desc="Creating Folds")):
        val_patient_ids = patient_ids[val_indices]
        
        fold_images = []
        for patient_id in val_patient_ids:
            fold_images.extend(patient_groups[patient_id])
            
        fold_path = os.path.join(ProjectPaths.det_dataset_org, f'fold_{i}.txt')
        with open(fold_path, 'w') as f:
            for image_name in sorted(fold_images):
                f.write(f"{image_name}\n")
        print(f"Saved Fold {i} with {len(val_patient_ids)} patients and {len(fold_images)} images to {fold_path}")

def generate_per_fold_annotations(k=5):
    """
    Uses the pre-generated fold files to create train and validation annotation
    files for each fold.
    """
    print(f"\n--- Generating Annotation Files for {k} Folds ---")
    all_fold_files = [os.path.join(ProjectPaths.det_dataset_org, f'fold_{i}.txt') for i in range(k)]
    
    fold_base_names = []
    for fold_file in all_fold_files:
        try:
            with open(fold_file, 'r') as f:
                fold_base_names.append(set(line.strip() for line in f))
        except FileNotFoundError:
            print(f"Error: Fold file not found: {fold_file}. Please run create_k_folds_by_patient() first.")
            return

    try:
        with open(ProjectPaths.det_annotations, 'r') as f:
            all_aug_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Augmented annotations not found: {ProjectPaths.det_annotations}. Please run augmentation.py first.")
        return

    for i in range(k):
        val_fold_names = fold_base_names[i]
        train_fold_names = set().union(*(fold_base_names[j] for j in range(k) if i != j))
        
        train_ann_path = os.path.join(ProjectPaths.det_dataset, f'train_annotations_fold_{i}.txt')
        val_ann_path = os.path.join(ProjectPaths.det_dataset, f'val_annotations_fold_{i}.txt')

        with open(train_ann_path, 'w') as train_f, open(val_ann_path, 'w') as val_f:
            for line in all_aug_lines:
                if not line.strip(): continue
                
                image_name_from_file = line.split()[0]
                base_name, aug_suffix = parse_image_name(image_name_from_file)
                
                pair_equivalent_name = base_name.replace('_DM_', '_CM_') if '_DM_' in base_name else base_name.replace('_CM_', '_DM_')
                
                is_in_train = base_name in train_fold_names or pair_equivalent_name in train_fold_names
                is_in_val = base_name in val_fold_names or pair_equivalent_name in val_fold_names

                # if is_in_train and aug_suffix != '_resized':
                if is_in_train:
                    train_f.write(line)
                elif is_in_val and aug_suffix == '_resized':
                    val_f.write(line)
        
        print(f"Generated {train_ann_path} and {val_ann_path}")

if __name__ == "__main__":
    # Step 1: Create the k-fold splits based on patient ID.
    # create_k_folds_by_patient(k=5)
    
    # Step 2: Generate the corresponding training and validation annotation files for each fold.
    generate_per_fold_annotations(k=5)

    print("\nK-fold dataset preparation complete.")
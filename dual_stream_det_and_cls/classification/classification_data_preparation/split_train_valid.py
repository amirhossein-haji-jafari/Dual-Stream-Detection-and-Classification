"""
train_annotations.txt will contain all augmented images with matching base name form augmented_annotations.txt.
For example train_annotations.txt should contain all of the bellow:
    P1_L_CM_MLO_blur, P1_L_CM_MLO_hflip, P1_L_CM_MLO_hvflip, P1_L_CM_MLO_noise, P1_L_CM_MLO_resized, P1_L_CM_MLO_rotate20, P1_L_CM_MLO_vflip
For each image only either CM or DM image should be in train_annotations.txt not both (as in train.txt only P1_L_CM_MLO is present and not P1_L_DM_MLO).
Also val_annotations.txt should only contain '_resized' image. Because we don't want to use augmentation for validation and testing.
and just like train_annotations.txt, either CM or DM image name should be in val_annotations.txt and test_annotations.txt.
"""

from ...immutables import ProjectPaths

import pandas as pd
import numpy as np

AUGMENTATION_SUFFIXES = [
    '_resized',
    "_hflip_noise",
    "_hvflip_brightcont",
    "_symm_shit_blur",
    '_rotate20_elastic',
    "_gridshuffle",
    "_cutout",
    # '_rotate20',
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

def main():
    # Read the train, validation, and test sets of exact image names
    try:
        with open(ProjectPaths.cls_dataset_org + '/train.txt', 'r') as f:
            train_set = set(line.strip() for line in f)

        with open(ProjectPaths.cls_dataset_org + '/val.txt', 'r') as f:
            val_set = set(line.strip() for line in f)
        
    except FileNotFoundError as e:
        print(f"Could not find split file {e.filename}.")
        print("Running splitting function first to generate train.txt and val.txt ...")
        train_val_split()
        try:
            with open(ProjectPaths.cls_dataset_org + '/train.txt', 'r') as f:
                train_set = set(line.strip() for line in f)

            with open(ProjectPaths.cls_dataset_org + '/val.txt', 'r') as f:
                val_set = set(line.strip() for line in f)
        except FileNotFoundError as e:
            print(f"Error: Could not find split file {e.filename} even after attempting to create it. Path may be incorrect.")
            return

    
    # Open the output files
    with open(ProjectPaths.cls_dataset + '/train_annotations_augmented.txt', 'w') as train_ann, \
         open(ProjectPaths.cls_dataset + '/val_annotations.txt', 'w') as val_ann: 
        
        print("Processing annotation files...")
        try:
            with open(ProjectPaths.cls_annotations, 'r') as annotations:
                for line in annotations:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    image_name_from_file = line.split()[0]
                    # Skip if image is DM. We assume that ProjectPaths.cls_annotations contains both CM and DM images.
                    if '_DM_' in image_name_from_file:
                        continue
                    base_name, aug_suffix = parse_image_name(image_name_from_file)
                    
                    # Generate the pair-equivalent name to handle mismatches
                    pair_equivalent_name = None
                    if '_DM_' in base_name:
                        pair_equivalent_name = base_name.replace('_DM_', '_CM_')
                    elif '_CM_' in base_name:
                        pair_equivalent_name = base_name.replace('_CM_', '_DM_')

                    # --- Logic for TRAIN set ---
                    # Include if the base name (e.g., P1_L_CM_MLO) is in train_set.
                    # This includes the original and ALL its augmentations, but only
                    # for the specific CM/DM version listed in train.txt.
                    # Include if the base name OR its pair-equivalent is in train_set.
                    if base_name in train_set or (pair_equivalent_name and pair_equivalent_name in train_set):
                        # if aug_suffix != "_resized":  # Exclude resized images from training set
                            train_ann.write(line)
                        
                    # --- Logic for VALIDATION set ---
                    # Include only if the base name OR its pair-equivalent is in val_set AND it's a resized image.
                    elif base_name in val_set or (pair_equivalent_name and pair_equivalent_name in val_set):
                        if aug_suffix == '_resized':
                            val_ann.write(line)

        except FileNotFoundError:
            print(f"Warning: Annotation file '{ProjectPaths.cls_annotations}' not found. Skipping.")

    print("Finished creating train, validation, and test annotation files with specified rules.")


def check_distribution(df, label_col='Pathology Classification/ Follow up'):
    """Check the distribution of classes in a DataFrame column."""
    dist = df[label_col].value_counts(normalize=True)
    return {k: f"{v*100:.2f}%" for k, v in dist.items()}

def train_val_split(train_size=0.8):
    # Read dataset
    annotations = pd.read_csv(ProjectPaths.annotations_consistent_harmonized)
    # filter annotations to only include either CM or DM images
    annotations = annotations[annotations['Image_name'].str.contains('_CM_')]
    # TODO: print 'Pathology Classification/ Follow up' distribution
    class_dist = check_distribution(annotations)
    print("class distribution: ",class_dist)
    # TODO: print number of images of each class
    print(annotations['Pathology Classification/ Follow up'].value_counts())

    # Group by patient_id
    grouped = annotations.groupby('Patient_ID')

    train_data = []
    valid_data = []

    for _, group in grouped:
        r = np.random.random()
        if r < train_size:
            train_data.append(group)
        else: # r < 1 - train_size
            valid_data.append(group)

    train_df = pd.concat(train_data)
    val_df = pd.concat(valid_data)

    train_dist = check_distribution(train_df)
    val_dist = check_distribution(val_df)

    # save distributions to a text file
    with open(ProjectPaths.cls_dataset_org + "/distributions.txt", 'w') as f:
        f.write(f"Class distributions: {class_dist}\n")
        f.write(f"Training set distribution: {train_dist}\n")
        f.write(f"Validation set distribution: {val_dist}\n")


    print(f"Training set distribution: {train_dist}")
    print(f"Validation set distribution: {val_dist}")

    # Save Only Image_name column to .txt files
    train_df['Image_name'].to_csv(ProjectPaths.cls_dataset_org + '/train.txt', index=False, header=False)
    val_df['Image_name'].to_csv(ProjectPaths.cls_dataset_org + '/val.txt', index=False, header=False)

    return None

if __name__ == "__main__":
    main()
    # train_val_split()

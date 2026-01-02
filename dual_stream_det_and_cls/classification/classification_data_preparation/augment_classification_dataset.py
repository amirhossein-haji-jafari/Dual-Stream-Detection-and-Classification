
"""
Data Augmentation Script for Paired Mammography Images for Classification.

This script performs a series of data augmentations on paired low-energy (DM)
and contrast-enhanced (CM) mammography images. The primary goal is to enrich the
dataset for an image-level classification task, making the training process 
more robust and effective.

The augmentation strategy is mirrored from the detection task's augmentation
script (`augment_detection_dataset.py`) to maintain consistency, but adapted
for a classification context (i.e., no bounding box transformations).

Key Concepts:
- Image Pair: A low-energy (DM) image and its corresponding contrast-enhanced
  (CM) image.
- Pairwise Transformation: A geometric operation (e.g., rotation, flip) that is
  applied identically to both images in a pair to maintain their spatial
  correspondence. This is handled using Albumentations' ReplayCompose.
- Non-Pairwise Transformation: A pixel-level operation (e.g., blur, brightness)
  that is applied independently to each image in the pair.

Augmentation Strategy:
1.  Resizing:
    - All images are resized to a uniform dimension of 1280x1280 pixels using a
      "letterbox" algorithm to preserve the aspect ratio by padding.

2.  Geometric & Pixel-Level Transformations:
    - The script applies a set of predefined, combined augmentation pipelines
      to generate diverse training samples. Each pipeline is applied to the
      resized original image pair.
    - Examples of pipelines include:
        - Rotation combined with Elastic Transformation.
        - Horizontal/Vertical flips combined with Brightness/Contrast adjustments.
        - A symmetric shift combined with Gaussian Blur.
        - Horizontal flip combined with Gaussian Noise.

3.  Output:
    - All generated images (resized originals and augmented versions) are saved
      to the classification dataset directory.
    - A corresponding annotation file (`cls_annotations.txt`) is created,
      listing each new image's filename and its image-level label (0 for Benign,
      1 for Malignant).

Input Data Structure:
- DM Images: From `ProjectPaths.low_energy_images_aligned`.
- CM Images: From `ProjectPaths.subtracted_images_aligned`.
- Labels: From `ProjectPaths.annotations_all_sheet_modified`, which contains
  image-level pathology classifications.
"""
import os
import cv2
import pandas as pd
import albumentations as A
from tqdm import tqdm
import shutil
from ...immutables import ProjectPaths

# --- Configuration ---
OUTPUT_IMG_DIR = ProjectPaths.cls_dataset
OUTPUT_ANN_FILE = os.path.join(ProjectPaths.cls_dataset, "augmented_cls_annotations.txt")

# Define image dimensions for resizing
TARGET_HEIGHT = 1280
TARGET_WIDTH = 1280
OUTPUT_EXTENSION = ".jpg"

# --- Helper Functions ---

def setup_directories():
    """Creates the necessary output directories, cleaning them first if they exist."""
    if os.path.exists(ProjectPaths.cls_dataset):
        shutil.rmtree(ProjectPaths.cls_dataset)
    os.makedirs(OUTPUT_IMG_DIR)
    print(f"Created output directory: {ProjectPaths.cls_dataset}")

def load_image_pairs_and_labels():
    """
    Scans image directories and annotation file to create a list of paired images with their labels.
    """
    annotations_df = pd.read_csv(ProjectPaths.annotations_consistent_harmonized)
    
    # Create a dictionary for quick label lookup
    # The 'Image_name' in the CSV does not have an extension
    label_dict = annotations_df.set_index('Image_name')['Pathology Classification/ Follow up'].to_dict()
    
    label_map = {
        "Benign": 0,
        "Malignant": 1,
        "Normal": 2,  # Special label for normal cases
    }

    image_pairs = []
    dm_dir = ProjectPaths.low_energy_images_aligned
    cm_dir = ProjectPaths.subtracted_images_aligned

    dm_files = [f for f in os.listdir(dm_dir) if f.endswith('.jpg') and '_DM_' in f]

    for dm_filename in tqdm(dm_files, desc="Finding image pairs"):
        base_name, _ = os.path.splitext(dm_filename)
        cm_filename = dm_filename.replace('_DM_', '_CM_')
        
        dm_path = os.path.join(dm_dir, dm_filename)
        cm_path = os.path.join(cm_dir, cm_filename)

        # Check if the pair exists
        if not os.path.exists(cm_path):
            print(f"Warning: Could not find CM pair for {dm_filename}. Skipping.")
            continue
        
        # Get the label using the CM name convention from the CSV
        cm_base_name = base_name.replace('_DM_', '_CM_')
        pathology = label_dict.get(cm_base_name)
        
        if pathology is None:
            print(f"Warning: No label found for {cm_base_name}. Skipping pair.")
            continue
            
        label = label_map.get(pathology)
        
        if label is None:
            print(f"Warning: Unknown pathology '{pathology}' for {cm_base_name}. Skipping.")
            continue
            
        # if label == -1: # Skip 'Normal' cases as they are not used for Benign/Malignant classification
        #      continue

        image_pairs.append({
            'dm_path': dm_path,
            'cm_path': cm_path,
            'label': label,
            'base_dm_name': base_name,
            'base_cm_name': cm_base_name
        })

    print(f"Successfully found {len(image_pairs)} image pairs for augmentation.")
    return image_pairs

def format_annotation_line(image_name, label):
    """Formats a single line for the output classification annotation file."""
    return f"{image_name} {label}\n"

# --- Main Augmentation Logic ---

def main():
    """Main script execution."""
    setup_directories()
    
    image_pairs = load_image_pairs_and_labels()

    # --- Define Augmentations (adapted from augment_detection_dataset.py) ---
    # No bbox_params are needed for this classification task.
    
    letterbox_transform = A.Compose([
        A.LongestMaxSize(max_size=max(TARGET_HEIGHT, TARGET_WIDTH), p=1.0),
        A.PadIfNeeded(
            min_height=TARGET_HEIGHT, 
            min_width=TARGET_WIDTH, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=(0, 0, 0),
            p=1.0
        ),
    ])

    # Replicating the augmentations from the detection script.
    augmentations = {
        'rotate20_elastic': A.ReplayCompose([
            A.Rotate(limit=(-20,20), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(0,0,0)),
            A.ElasticTransform(p=1.0, alpha=300, sigma=10)
        ]),
        # 'rotate20': A.ReplayCompose([
        #     A.Rotate(limit=(-20,20), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(0,0,0)),
        #     # A.ElasticTransform(p=1.0, alpha=300, sigma=10)
        # ]),
        'hvflip_brightcont': A.Compose([
            A.HorizontalFlip(p=1.0), 
            A.VerticalFlip(p=1.0), 
            A.RandomBrightnessContrast(p=1.0)
        ]),
        'symm_shit_blur': A.ReplayCompose([
            A.GaussianBlur(blur_limit=(2, 7), p=1.0),
            A.Affine(translate_percent={'x': 0.2, 'y':  0.05}, p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(0, 0, 0))
        ]),
        'hflip_noise': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.GaussNoise(std_range=[0.05, 0.1], mean_range=[0, 0],per_channel=False ,noise_scale_factor=1)
        ]),
        'gridshuffle': A.ReplayCompose([
            A.RandomGridShuffle(grid=(3,3), p=1.0),
        ]),
        'cutout': A.Compose([A.CoarseDropout(num_holes_range=(1,3), hole_height_range=(0.05, 0.15), hole_width_range=(0.1, 0.2), fill=0, p=1.0)]),
    }
    
    new_annotations = []

    for data in tqdm(image_pairs, desc="Augmenting Pairs"):
        dm_image = cv2.imread(data['dm_path'], cv2.IMREAD_COLOR)
        cm_image = cv2.imread(data['cm_path'], cv2.IMREAD_COLOR)
        
        if dm_image is None or cm_image is None:
            print(f"Warning: Could not read images for pair based on {data['base_dm_name']}. Skipping.")
            continue
            
        # --- Resize original images ---
        resized_dm_image = letterbox_transform(image=dm_image)['image']
        resized_cm_image = letterbox_transform(image=cm_image)['image']

        # --- Save Original Resized Images ---
        base_dm_name = data['base_dm_name'] + "_resized" + OUTPUT_EXTENSION
        base_cm_name = data['base_cm_name'] + "_resized" + OUTPUT_EXTENSION
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, base_dm_name), resized_dm_image)
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, base_cm_name), resized_cm_image)
        
        # Add both to the annotation file
        new_annotations.append(format_annotation_line(base_dm_name, data['label']))
        new_annotations.append(format_annotation_line(base_cm_name, data['label']))

        # --- Augmentations Loop ---
        for aug_name, aug_pipeline in augmentations.items():
            
            aug_dm_image = None
            aug_cm_image = None
            
            # Use ReplayCompose for pairwise geometric augmentations to ensure they are identical
            if isinstance(aug_pipeline, A.ReplayCompose):
                aug_dm_data = aug_pipeline(image=resized_dm_image)
                aug_cm_data = A.ReplayCompose.replay(aug_dm_data['replay'], image=resized_cm_image)
                aug_dm_image = aug_dm_data['image']
                aug_cm_image = aug_cm_data['image']
            else: # For Compose (can be geometric or pixel-level)
                  # Pixel-level transforms are applied independently, which is desired.
                  # Deterministic geometric transforms (p=1.0) are applied identically.
                aug_dm_image = aug_pipeline(image=resized_dm_image)['image']
                aug_cm_image = aug_pipeline(image=resized_cm_image)['image']

            aug_dm_name = f"{data['base_dm_name']}_{aug_name}{OUTPUT_EXTENSION}"
            aug_cm_name = f"{data['base_cm_name']}_{aug_name}{OUTPUT_EXTENSION}"
            cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, aug_dm_name), aug_dm_image)
            cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, aug_cm_name), aug_cm_image)

            # Add both augmented images to the annotation file
            new_annotations.append(format_annotation_line(aug_dm_name, data['label']))
            new_annotations.append(format_annotation_line(aug_cm_name, data['label']))
            
    with open(OUTPUT_ANN_FILE, 'w') as f:
        f.writelines(new_annotations)
    
    total_images = len(new_annotations)
    print("\n--- Augmentation Complete ---")
    print(f"Saved {total_images} new images to: {OUTPUT_IMG_DIR}")
    print(f"Saved new annotations to: {OUTPUT_ANN_FILE}")


if __name__ == '__main__':
    main()
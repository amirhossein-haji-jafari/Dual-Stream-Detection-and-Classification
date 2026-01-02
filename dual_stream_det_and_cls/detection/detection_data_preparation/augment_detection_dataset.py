"""
Data Augmentation Script for Paired Mammography Images.

This script performs a series of data augmentations on paired low-energy (DM)
and contrast-enhanced (CM) mammography images. The primary goal is to enrich the
dataset, making the training process more robust and effective, particularly
against minor misalignments between image pairs.

Key Concepts:
- Fixed Image: The reference image whose filename is listed in the annotation
  file (e.g., 'P1_L_DM_MLO'). Bounding boxes are defined relative to this
  image. The `data['original_filename']` variable holds this name.
- Moving Image: The corresponding image in the pair, whose filename is inferred
  by string replacement (e.g., swapping 'DM' for 'CM'). This image is subject
  to a special, non-pairwise shift transformation to simulate misalignment.
- Pairwise Transformation: A geometric operation (e.g., rotation, flip) that is
  applied identically to both the fixed and moving images to maintain their
  spatial correspondence. Bounding boxes are also transformed accordingly.
- Non-Pairwise Transformation: A pixel-level operation (e.g., blur, brightness)
  that is applied independently to the contrast-enhanced (subtracted) and low-energy images.

Augmentation Strategy:
The augmentation pipeline is designed with specific rules for geometric and
pixel-level transformations to create challenging and realistic training samples.

1.  Resizing:
    - All images (both low-energy and contrast-enhanced) are resized to a uniform dimension of
      1280x1280 pixels.
    - A "letterbox" algorithm is used to preserve the original aspect ratio by
      padding the image as necessary.

2.  Geometric Transformations (Pairwise):
    - Random Rotation: Rotates the image pair by a random angle up to
          +/- 20 degrees.
    - Horizontal flip
    - Horizontal-vertical flip
    - Elastic Transformation: Simulates tissue deformation.
    - Grid Shuffle: Divides the image into a grid and shuffles the grid cells. 
        A more extreme augmentation that can help models learn texture rather than high-level spatial arrangements.

3.  Misalignment Simulation (Special Case):
    - A small shift transformation (maximum of 5 pixels) is applied ONLY to
      the `moving_image`.
    - The `fixed_image` and its bounding box annotations remain unchanged by
      this specific transformation.
    - This is crucial for training a model that is robust to minor registration
      errors between DM and CM scans.

4.  Pixel-Level Transformations (Non-Pairwise):
    - These transformations are applied independently to each image in the pair.
    - Examples include:
        - Blur (e.g., Gaussian Blur)
        - Random Brightness and Contrast

To enhance epoch differentiation and prevent feeding redundant data to network, this project uses specific combinations of transformations:
    - Horizontal-vertical flip and Random Brightness and Contrast
    - Cutout (Coarse Dropout) and horizontal flip
    - Shift and blurring
    - Rotate and Elastic Transformation
    - Grid Shuffle


Lastly we save the newly augmented images and the corresponding transformed bounding
    box annotations to new files using a clear and descriptive naming scheme.

Input Data Structure:
- Image Pairs: The script expects pairs of DM and CM images.
- Annotation File (`annotation.txt`):
    - This file contains annotations ONLY for the `fixed_image` of each pair.
    - The filename of the `moving_image` is not present and must be inferred.
    - An image can have multiple bounding boxes
    - An example of annotations:
        P5_L_CM_MLO.jpg 1 95,125,250,300,0 350,80,450,180,0
            Column 1: image name.
            Column 2: Image-level label (1 for Malignant, 0 for Benign).
            Subsequent Columns: Bounding boxes in min_x,min_y,max_x,max_y,class_id format. 
                Since we only have one object class ("mass"), the class_id is always 0.
"""
"""
Other Suggested Conventional Medical Image Augmentations (Apply if DualStreamRetinaNet is not perfoming well):
    Grid Distortion (A.GridDistortion): Creates complex, non-linear distortions in the image.
"""
import os
import cv2
import albumentations as A
from tqdm import tqdm
import re
import shutil
from ...immutables import ProjectPaths, Hyperparameter
from ...medical_image_utils import clahe

# --- Configuration ---
# Define paths relative to the script's location
OUTPUT_IMG_DIR = os.path.join(ProjectPaths.det_dataset, "")
# OUTPUT_ANN_FILE = os.path.join(ProjectPaths.det_annotations, "")
OUTPUT_ANN_FILE = ProjectPaths.det_annotations

# Define image dimensions for resizing
TARGET_HEIGHT = 1280
TARGET_WIDTH = 1280
OUTPUT_EXTENSION = ".jpg"
# --- Helper Functions ---

def setup_directories():
    """Creates the necessary output directories, cleaning them first if they exist."""
    if os.path.exists(ProjectPaths.det_dataset):
        shutil.rmtree(ProjectPaths.det_dataset)
    os.makedirs(OUTPUT_IMG_DIR)
    print(f"Created output directory: {ProjectPaths.det_dataset}")

def get_pair_key(filename):
    """
    Generates a unique key for a DM/CM pair.
    Example: 'P1_L_DM_MLO.jpg' -> 'P1_L_MLO'
    Example: 'P2_R_CM_CC.png' -> 'P2_R_CC'
    """
    # Remove the DM/CM part and the extension
    key = re.sub(r'_(DM|CM)_', '_', filename)
    key = os.path.splitext(key)[0]
    return key

def parse_annotations(ann_file_path, image_dir):
    """
    Parses the annotation file and groups data by image pairs.
    Returns a dictionary where each key represents an image pair.
    """
    if not os.path.exists(ann_file_path):
        raise FileNotFoundError(f"Annotation file not found: {ann_file_path}")

    paired_data = {}
    with open(ann_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        
        parts = line.strip().split()
        filename_with_ext = parts[0]
        filename_base, _ = os.path.splitext(filename_with_ext)
        
        label = int(parts[1])
        bboxes = []
        if len(parts) > 2:
            for bbox_str in parts[2:]:
                coords = [int(c) for c in bbox_str.split(',')]
                # Bbox: [x_min, y_min, x_max, y_max, class_id]
                bboxes.append(coords)

        pair_key = get_pair_key(filename_with_ext)

        if pair_key not in paired_data:
            # Find the corresponding DM and CM files by replacing the token
            dm_filename_base = filename_base.replace('_CM_', '_DM_')
            cm_filename_base = filename_base.replace('_DM_', '_CM_')
            
            # Assuming all input images are jpg as per example
            dm_path = os.path.join(image_dir, dm_filename_base + ".jpg")
            cm_path = os.path.join(image_dir, cm_filename_base + ".jpg")
        
            if not (os.path.exists(dm_path) and os.path.exists(cm_path)):
                print(f"Warning: Could not find pair for {filename_with_ext}. Skipping.")
                continue

            paired_data[pair_key] = {
                'dm_path': dm_path,
                'cm_path': cm_path,
                'label': label,
                'bboxes': bboxes,
                'base_dm_name': dm_filename_base,
                'base_cm_name': cm_filename_base,
                'original_filename': filename_with_ext
            }

    print(f"Successfully parsed {len(paired_data)} image pairs from annotations.")
    return paired_data

def format_annotation_line(image_name, label, bboxes):
    """Formats a single line for the output annotation file."""
    bbox_strs = ["{:.0f},{:.0f},{:.0f},{:.0f},{:d}".format(*b) for b in bboxes]
    return f"{image_name} {label} {' '.join(bbox_strs)}\n"

# --- Main Augmentation Logic ---

def main():
    """Main script execution."""
    setup_directories()
    
    paired_annotations = parse_annotations(
        os.path.join(ProjectPaths.det_dataset_org, "annotations.txt"),
        ProjectPaths.det_dataset_org
    )

    BBOX_PARAMS = A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.50)
    
    letterbox_transform = A.Compose([
        A.LongestMaxSize(max_size=max(TARGET_HEIGHT, TARGET_WIDTH), p=1.0),
        A.PadIfNeeded(
            min_height=TARGET_HEIGHT, 
            min_width=TARGET_WIDTH, 
            border_mode=cv2.BORDER_CONSTANT, 
            fill=(0, 0, 0),
            p=1.0
        ),
    ], bbox_params=BBOX_PARAMS)

    augmentations = {
        # 'hflip': A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=BBOX_PARAMS),
        # 'vflip': A.Compose([A.VerticalFlip(p=1.0)], bbox_params=BBOX_PARAMS),
        # 'hvflip': A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], bbox_params=BBOX_PARAMS),
        # 'rotate20': A.ReplayCompose([A.Rotate(limit=(-20,20), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(0,0,0))], bbox_params=BBOX_PARAMS),
        'rotate20_elastic': A.ReplayCompose([A.Rotate(limit=(-20,20), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(0,0,0)),
                                A.ElasticTransform(p=1.0, alpha=300, sigma=10, keypoint_remapping_method='mask')
                                ], bbox_params=BBOX_PARAMS),
        'rotate20': A.ReplayCompose([A.Rotate(limit=(-20,20), p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(0,0,0)),
                                ], bbox_params=BBOX_PARAMS),
        # 'blur': A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)]),
        # 'brightcont': A.Compose([A.RandomBrightnessContrast(p=1.0)]),
        'cutout': A.Compose([A.CoarseDropout(num_holes_range=(1,3), hole_height_range=(0.05, 0.15), hole_width_range=(0.1, 0.2), fill=0, p=1.0)]),
        # 'cutout_mulnoise': A.Compose([
        #     A.CoarseDropout(num_holes_range=(1,3), hole_height_range=(0.05, 0.15), hole_width_range=(0.1, 0.2), fill=0, p=1.0),
        #     A.MultiplicativeNoise(multiplier=[0.8, 1.4],per_channel=False,elementwise=True),
        #     ]),
        'hvflip_brightcont': A.Compose([
            A.HorizontalFlip(p=1.0), 
            A.VerticalFlip(p=1.0), 
            A.RandomBrightnessContrast(p=1.0)
        ], bbox_params=BBOX_PARAMS),
        # 'hflip_cutout': A.Compose([
        #     A.HorizontalFlip(p=1.0),
        #     A.CoarseDropout(num_holes_range=(1,3), hole_height_range=(0.05, 0.15), hole_width_range=(0.1, 0.2), fill=0, p=1.0)
        # ], bbox_params=BBOX_PARAMS),
        'gridshuffle': A.ReplayCompose([
            A.RandomGridShuffle(grid=(3,3), p=1.0),
        ], bbox_params=BBOX_PARAMS),
        'symm_shit_blur': A.ReplayCompose([
            A.GaussianBlur(blur_limit=(2, 7), p=1.0),
            A.Affine(translate_percent={'x': 0.2, 'y':  0.05}, p=1.0, border_mode=cv2.BORDER_CONSTANT, fill=(0, 0, 0))
        ], bbox_params=BBOX_PARAMS),
        'hflip_noise': A.Compose([
            A.HorizontalFlip(p=1.0),
            A.GaussNoise(std_range=[0.05, 0.1], mean_range=[0, 0],per_channel=False ,noise_scale_factor=1)
        ], bbox_params=BBOX_PARAMS),
    }

    new_annotations = []

    for pair_key, data in tqdm(paired_annotations.items(), desc="Augmenting Pairs"):
        dm_image = cv2.imread(data['dm_path'], cv2.IMREAD_COLOR)
        cm_image = cv2.imread(data['cm_path'], cv2.IMREAD_COLOR)
        if dm_image is None or cm_image is None:
            print(f"Warning: Could not read images for pair {pair_key}. Skipping.")
            continue
        
        if Hyperparameter.use_clahe:
            dm_image = clahe(dm_image)
            cm_image = clahe(cm_image)
            
        original_bboxes_coords = [b[:4] for b in data['bboxes']]
        original_labels = [b[4] for b in data['bboxes']]
        # --- Handle cases with no bounding boxes ---
        if not original_bboxes_coords:
            # If no bboxes, just resize and save the image without bbox processing
            # This avoids errors with albumentations on empty bbox lists
            resized_dm_image = letterbox_transform(image=dm_image)['image']
            resized_cm_image = letterbox_transform(image=cm_image)['image']
            resized_bboxes_to_save = []
        else:
            resized_dm_data = letterbox_transform(image=dm_image, bboxes=original_bboxes_coords, class_labels=original_labels)
            resized_cm_image = letterbox_transform(image=cm_image, bboxes=original_bboxes_coords, class_labels=original_labels)['image']
            
            resized_dm_image = resized_dm_data['image']
            resized_bboxes_coords = resized_dm_data['bboxes']
            # Re-combine the transformed coordinates with their original labels
            resized_bboxes_to_save = [list(coords) + [label] for coords, label in zip(resized_bboxes_coords, original_labels)]

        # --- Save Original Resized Images ---
        base_dm_name = data['base_dm_name'] + "_resized" + OUTPUT_EXTENSION
        base_cm_name = data['base_cm_name'] + "_resized" + OUTPUT_EXTENSION


        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, base_dm_name), resized_dm_image)
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, base_cm_name), resized_cm_image)
        # Only write annotation for the original image type
        if '_DM_' in data['original_filename']:
            new_annotations.append(format_annotation_line(base_dm_name, data['label'], resized_bboxes_to_save))
        else: # _CM_
            new_annotations.append(format_annotation_line(base_cm_name, data['label'], resized_bboxes_to_save))
        
        # Skip further augmentation if there were no boxes to augment
        if not original_bboxes_coords:
            continue

        # # --- Asymmetric Augmentation: Shift ---
        # # Apply shift only to the moving image
        # shift_transform = A.Affine(
        #     translate_px={'x': 5, 'y':  5}, 
        #     p=1.0, 
        #     border_mode=cv2.BORDER_CONSTANT, 
        #     fill=(0, 0, 0)
        # )
        
        # shifted_dm_image = resized_dm_image.copy()
        # shifted_cm_image = resized_cm_image.copy()
        # is_dm_fixed = '_DM_' in data['original_filename']
        
        # if is_dm_fixed:
        #     shifted_cm_image = shift_transform(image=shifted_cm_image)['image']
        # else:
        #     shifted_dm_image = shift_transform(image=shifted_dm_image)['image']
        
        # # BBoxes are from the fixed image, which was not shifted.
        # bboxes_for_shift_aug = resized_bboxes_to_save 
    
        # # --- Combined Asymmetric Augmentation: Shift then Blur ---
        # # We already have the shifted images. Now apply blur to both.
        # blur_transform = A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)])
        
        # blurred_shifted_dm_image = blur_transform(image=shifted_dm_image)['image']
        # blurred_shifted_cm_image = blur_transform(image=shifted_cm_image)['image']

        # # The bounding boxes remain the same as the simple shift augmentation
        # # because blur does not change geometry.
        # bboxes_for_shift_blur_aug = bboxes_for_shift_aug

        # shift_blur_dm_name = f"{data['base_dm_name']}_shift_blur{OUTPUT_EXTENSION}"
        # shift_blur_cm_name = f"{data['base_cm_name']}_shift_blur{OUTPUT_EXTENSION}"
        # cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, shift_blur_dm_name), blurred_shifted_dm_image)
        # cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, shift_blur_cm_name), blurred_shifted_cm_image)

        # if is_dm_fixed:
        #     new_annotations.append(format_annotation_line(shift_blur_dm_name, data['label'], bboxes_for_shift_blur_aug))
        # else:
        #     new_annotations.append(format_annotation_line(shift_blur_cm_name, data['label'], bboxes_for_shift_blur_aug))

        # --- Symmetric Augmentations Loop ---
        for aug_name, aug_pipeline in augmentations.items():
            # The coordinates to be augmented are from the resized image
            resized_coords = [b[:4] for b in resized_bboxes_to_save]

            # Use ReplayCompose for pairwise random augmentations
            if isinstance(aug_pipeline, A.ReplayCompose):
                # Apply to DM first to get replay data
                aug_dm_data = aug_pipeline(image=resized_dm_image, bboxes=resized_coords, class_labels=original_labels)
                # Re-apply the same transform to CM using replay data
                aug_cm_data = A.ReplayCompose.replay(aug_dm_data['replay'], image=resized_cm_image, bboxes=resized_coords, class_labels=original_labels)
                aug_dm_image = aug_dm_data['image']
                aug_cm_image = aug_cm_data['image']
                aug_bboxes_coords = aug_dm_data['bboxes']
            else: # logic for deterministic and non-geometric transforms
                # Check if the augmentation affects bounding boxes
                is_geometric = 'bbox_params' in aug_pipeline.get_dict_with_id()
                if is_geometric:
                    # Apply the same geometric transform to both images and the bboxes once
                    aug_dm_data = aug_pipeline(image=resized_dm_image, bboxes=resized_coords, class_labels=original_labels)
                    aug_cm_data = aug_pipeline(image=resized_cm_image, bboxes=resized_coords, class_labels=original_labels)
                    aug_dm_image = aug_dm_data['image']
                    aug_cm_image = aug_cm_data['image']
                    aug_bboxes_coords = aug_dm_data['bboxes']
                else: # Non-geometric (pixel-level) transform
                    # Apply to each image independently, bboxes are unchanged
                    aug_dm_image = aug_pipeline(image=resized_dm_image)['image']
                    aug_cm_image = aug_pipeline(image=resized_cm_image)['image']
                    aug_bboxes_coords = resized_coords

            # Re-combine the augmented coordinates with their original labels
            aug_bboxes_to_save = [list(coords) + [label] for coords, label in zip(aug_bboxes_coords, original_labels)]

            aug_dm_name = f"{data['base_dm_name']}_{aug_name}{OUTPUT_EXTENSION}"
            aug_cm_name = f"{data['base_cm_name']}_{aug_name}{OUTPUT_EXTENSION}"
                
            cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, aug_dm_name), aug_dm_image)
            cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, aug_cm_name), aug_cm_image)

            # Only write annotation for the original image type
            if '_DM_' in data['original_filename']:
                new_annotations.append(format_annotation_line(aug_dm_name, data['label'], aug_bboxes_to_save))
            else: # _CM_
                new_annotations.append(format_annotation_line(aug_cm_name, data['label'], aug_bboxes_to_save))
            
    with open(OUTPUT_ANN_FILE, 'w') as f:
        f.writelines(new_annotations)
    
    print("\n--- Augmentation Complete ---")
    print(f"Saved {len(os.listdir(OUTPUT_IMG_DIR))} new images to: {OUTPUT_IMG_DIR}")
    print(f"Saved new annotations to: {OUTPUT_ANN_FILE}")


if __name__ == '__main__':
    main()
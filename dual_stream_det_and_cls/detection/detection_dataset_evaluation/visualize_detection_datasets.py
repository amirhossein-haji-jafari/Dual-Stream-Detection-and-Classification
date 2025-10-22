"""
We want to visually evaluate images that are used for training, validating and testing (from mass_u_enhancement_u_nme):
	1- Original fixed image and all of it's segmentation masks. Image comes from either 'final-project\dataset\Low energy images of CDD-CESM' or 'final-project\dataset\Subtracted images of CDD-CESM'.
        Segmentation mask comes from Radiology_hand_drawn_segmentations_v2.csv.
	1a- Aligned fixed and registered images with bounding box annotaions. Images and annotations should come from 'final-project\dual_stream_RetinaNet\dataset_org'.
	2- All of augmented low-energy (DM) and their bounding boxes.
	3- All of augmented contrast-enhanced (CM) images and their bounding boxes. corresponding augmented DM and CM images should be below one each other.
        Augmented images and annotations should come from 'final-project\dual_stream_RetinaNet\dataset'.
Original fixed image label comes from 'final-project\dataset\Radiology_manual_annotations_all_sheet_modified.csv'.
This label should match with label of all other images that we want to visually evaluate.
The color of segmentation masks and bounding boxes will be green for 'Benign' cases and red otherwise.
lastly, we save all images in one .png image with three rows.
"""


import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Project Imports ---
# Assuming the script is run from a location where this path is valid
from immutables import ProjectPaths
from ...medical_image_utils import get_segmented_image, alpha_blend
from detection_data_preparation.detection_dataset_definitions import mass_union_mass_enhancement, aligned_image_names


# Define the suffixes for the different augmentations
AUGMENTATION_SUFFIXES = [
    '_resized',
    '_rotate20',
    "_hvflip_brightcont",
    "_symm_shit_blur",
    "_hflip_noise",
    # '_rotate20_elastic',
    # '_hflip',
    # '_vflip',
    # '_hvflip', 
    # '_blur',
    # '_brightcont',
    # '_cutout',
    # '_shift',
    # "_hflip_cutout",
    # "_shift_blur",
    # "_gridshuffle",
]

# --- Helper Functions ---

def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=4):
    """Draws bounding boxes on a copy of the image."""
    img_copy = image.copy()
    if bboxes is None:
        return img_copy
    for bbox in bboxes:
        # Format is [x_min, y_min, x_max, y_max, class_id]
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)
    return img_copy

def parse_annotations(ann_file_path):
    """Parses an annotation file into a dictionary."""
    annotations = {}
    if not os.path.exists(ann_file_path):
        print(f"Warning: Annotation file not found: {ann_file_path}")
        return annotations
        
    with open(ann_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            image_name = parts[0]
            label = int(parts[1])
            bboxes = []
            if len(parts) > 2:
                for bbox_str in parts[2:]:
                    coords = [int(c) for c in bbox_str.split(',')]
                    bboxes.append(coords)
            annotations[image_name] = {'label': label, 'bboxes': bboxes}
            
    return annotations

def get_augmented_files(base_name, aug_dir):
    """Finds all augmented files for a given base name."""
    augs = {}
    for suffix in AUGMENTATION_SUFFIXES:
        fname_with_ext = f"{base_name}{suffix}.jpg"
        if os.path.exists(os.path.join(aug_dir, fname_with_ext)):
             augs[suffix] = fname_with_ext
    return augs
    
def main():
    if not os.path.exists(ProjectPaths.visualize_det_datasets):
        os.makedirs(ProjectPaths.visualize_det_datasets)
        print(f"Created output directory: {ProjectPaths.visualize_det_datasets}")

    # Load necessary data
    segmentations_df = pd.read_csv(ProjectPaths.segmentations)
    org_annotations = parse_annotations(ProjectPaths.det_annotations_org)
    aug_annotations = parse_annotations(ProjectPaths.det_annotations)
    annotations_df = pd.read_csv(ProjectPaths.annotations_all_sheet_modified)
    label_lookup = pd.Series(
        annotations_df['Pathology Classification/ Follow up'].values, 
        index=annotations_df['Image_name']
    ).to_dict()

    # --- Main Loop ---
    for cm_image_base in tqdm(mass_union_mass_enhancement, desc="Processing Image Pairs"):
        dm_image_base = cm_image_base.replace("_CM_", "_DM_")
        pair_key = dm_image_base.replace("_DM_", "_")

        # --- 1. Gather data for Row 1 ---
        # 1a. Original Fixed Image + Segmentation Mask and its label
        is_cm_fixed = cm_image_base not in aligned_image_names
        fixed_image_name = cm_image_base if is_cm_fixed else dm_image_base
        label = label_lookup.get(fixed_image_name, "Unknown")
        
        # Set color based on the label
        if label == "Benign":
            draw_color_bgr = (0, 255, 0)  # Green in BGR
        else:
            draw_color_bgr = (0, 0, 255)  # Red in BGR

        # --- 1. Gather data for Row 1 ---
        # 1a. Original Fixed Image + Segmentation Mask
        original_image_path = os.path.join(
            ProjectPaths.subtracted_images if is_cm_fixed else ProjectPaths.low_energy_images,
            f"{fixed_image_name}.jpg"
        )
        original_image = cv2.imread(original_image_path)
        if original_image is None:
            print(f"Warning: Could not load original image {original_image_path}. Skipping pair {pair_key}.")
            continue
        
        masks = segmentations_df[segmentations_df["#filename"] == f"{fixed_image_name}.jpg"]["region_shape_attributes"]
        if masks.empty:
            blended_img = original_image.copy() # No mask to blend
        else:
            seg_mask = np.array(get_segmented_image(original_image, masks)) > 0
            blended_img = alpha_blend(original_image.copy(), seg_mask, color_bgr=draw_color_bgr)
        
        # 1b. Aligned DM/CM Images + BBoxes
        aligned_dm_path = os.path.join(ProjectPaths.det_dataset_org, f"{dm_image_base}.jpg")
        aligned_cm_path = os.path.join(ProjectPaths.det_dataset_org, f"{cm_image_base}.jpg")
        aligned_dm_img = cv2.imread(aligned_dm_path)
        aligned_cm_img = cv2.imread(aligned_cm_path)

        if aligned_dm_img is None or aligned_cm_img is None:
            print(f"Warning: Could not load aligned images for {pair_key}. Skipping.")
            continue
            
        # Get BBoxes - they are the same for the pair, but file could be keyed by DM or CM
        org_bboxes = org_annotations.get(f"{dm_image_base}", {}).get('bboxes')
        if org_bboxes is None:
             org_bboxes = org_annotations.get(f"{cm_image_base}", {}).get('bboxes')

        aligned_dm_with_boxes = draw_bboxes(aligned_dm_img, org_bboxes, color=draw_color_bgr)
        aligned_cm_with_boxes = draw_bboxes(aligned_cm_img, org_bboxes, color=draw_color_bgr)

        # --- 2. Gather data for Rows 2 & 3 ---
        dm_augs = get_augmented_files(dm_image_base, ProjectPaths.det_dataset)
        cm_augs = get_augmented_files(cm_image_base, ProjectPaths.det_dataset)
        
        # Order by suffix for consistency
        aug_suffixes_ordered = sorted(dm_augs.keys(), key=lambda x: AUGMENTATION_SUFFIXES.index(x) if x in AUGMENTATION_SUFFIXES else -1)
        num_cols = len(aug_suffixes_ordered)
        if num_cols == 0:
            print(f"No augmentations found for {pair_key}. Skipping.")
            continue

        # --- 3. Create the plot ---
        fig = plt.figure(figsize=(num_cols * 5, 15))
        gs = plt.GridSpec(3, num_cols, figure=fig)
        
        # Plot Row 1
        ax1_1 = fig.add_subplot(gs[0, 0:num_cols//3])
        ax1_2 = fig.add_subplot(gs[0, num_cols//3:2*num_cols//3])
        ax1_3 = fig.add_subplot(gs[0, 2*num_cols//3:])

        ax1_1.imshow(cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB))
        ax1_1.set_title(f"Original Fixed: {fixed_image_name}\n+ Seg Mask", fontsize=10)
        ax1_1.axis('off')

        ax1_2.imshow(cv2.cvtColor(aligned_dm_with_boxes, cv2.COLOR_BGR2RGB))
        ax1_2.set_title(f"Aligned DM: {dm_image_base}\n+ BBoxes", fontsize=10)
        ax1_2.axis('off')

        ax1_3.imshow(cv2.cvtColor(aligned_cm_with_boxes, cv2.COLOR_BGR2RGB))
        ax1_3.set_title(f"Aligned CM: {cm_image_base}\n+ BBoxes", fontsize=10)
        ax1_3.axis('off')

        # Plot Rows 2 and 3
        for i, suffix in enumerate(aug_suffixes_ordered):
            cm_aug_fname = cm_augs.get(suffix)
            dm_aug_fname = dm_augs.get(suffix)
            local_pair_key = ""
            if (cm_aug_fname is None) and dm_aug_fname:
                cm_aug_fname = dm_aug_fname.replace("_DM_", "_CM_")
            else:
                dm_aug_fname = cm_aug_fname.replace("_CM_", "_DM_")

            if cm_aug_fname in aug_annotations:
                local_pair_key = cm_aug_fname
            else:
                local_pair_key = dm_aug_fname
            # Row 2 (DM)
            ax_dm = fig.add_subplot(gs[1, i])
           
            dm_aug_path = os.path.join(ProjectPaths.det_dataset, dm_aug_fname)
            dm_img = cv2.imread(dm_aug_path)
            dm_bboxes = aug_annotations[local_pair_key]['bboxes']

            # Set color based on the label
            dm_aug_label = aug_annotations[local_pair_key]['label']
            if dm_aug_label == 0:
                draw_color_bgr = (0, 255, 0)  # Green in BGR
            else:
                draw_color_bgr = (0, 0, 255)  # Red in BGR

            dm_img_boxes = draw_bboxes(dm_img, dm_bboxes, color=draw_color_bgr)
            ax_dm.imshow(cv2.cvtColor(dm_img_boxes, cv2.COLOR_BGR2RGB))
            ax_dm.set_title(dm_aug_fname, fontsize=8)
            ax_dm.axis('off')
            
            # Row 3 (CM)
            ax_cm = fig.add_subplot(gs[2, i])
    
            cm_aug_path = os.path.join(ProjectPaths.det_dataset, cm_aug_fname)
            cm_img = cv2.imread(cm_aug_path)
            cm_bboxes = aug_annotations[local_pair_key]['bboxes']

            # Set color based on the label
            cm_aug_label = aug_annotations[local_pair_key]['label']
            if cm_aug_label == 0:
                draw_color_bgr = (0, 255, 0)
            else:
                draw_color_bgr = (0, 0, 255)

            cm_img_boxes = draw_bboxes(cm_img, cm_bboxes, color=draw_color_bgr)
            ax_cm.imshow(cv2.cvtColor(cm_img_boxes, cv2.COLOR_BGR2RGB))
            ax_cm.set_title(cm_aug_fname, fontsize=8)
            ax_cm.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(ProjectPaths.visualize_det_datasets, f"{pair_key}.png"), dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    main()
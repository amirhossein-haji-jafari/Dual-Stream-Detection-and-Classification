from ...immutables import ProjectPaths
from .detection_dataset_definitions import (
    mass_union_mass_enhancement,
    aligned_image_names,
    image_masks_to_exclude,
)
import pandas as pd
import cv2
from ..utils.medical_image_utils import get_bounding_boxes
from tqdm import tqdm  # Add tqdm for progress bar

# save_path = "/home/monstalinux/final-project/dual_stream_RetinaNet/dataset_org/"
save_path = ProjectPaths.det_dataset_org + "/"
segmentations = pd.read_csv(ProjectPaths.segmentations)
fixed_image_is_cm = False
lines = []
annotations = pd.read_csv(ProjectPaths.annotations_all_sheet_modified)

# Wrap the main loop with tqdm for progress visualization
for cm_image_name in tqdm(mass_union_mass_enhancement, desc="Processing images"):
    dm_image_name = cm_image_name.replace("CM", "DM")

    # figure which of CM or DM is fixed image.
    if cm_image_name not in aligned_image_names:
        fixed_image_is_cm = True
        # use CM image segmentation masks as detection annotations.
        masks = segmentations[segmentations["#filename"] == cm_image_name + ".jpg"]["region_shape_attributes"]
        # mask should not be empty
        if masks.empty:
            raise ValueError(f"mask for {cm_image_name} is empty.")
        # drop masks with matching index in image_masks_to_exclude
        if cm_image_name in image_masks_to_exclude:
            mask_indices_to_drop = image_masks_to_exclude[cm_image_name]
            masks = masks.drop(mask_indices_to_drop)
            bounding_boxes = get_bounding_boxes(masks)
        else:
            bounding_boxes = get_bounding_boxes(masks)
    elif cm_image_name in aligned_image_names:
        fixed_image_is_cm = False
        masks = segmentations[segmentations["#filename"] == dm_image_name + ".jpg"]["region_shape_attributes"]
        # mask should not be empty
        if masks.empty:
            raise ValueError(f"mask for {dm_image_name} is empty.")
        # drop masks with matching index in image_masks_to_exclude
        if dm_image_name in image_masks_to_exclude:
            mask_indices_to_drop = image_masks_to_exclude[dm_image_name]
            masks = masks.drop(mask_indices_to_drop)
            bounding_boxes = get_bounding_boxes(masks)
        else:
            bounding_boxes = get_bounding_boxes(masks)

    # path of aligned images
    cm_image_path = ProjectPaths.subtracted_images_aligned + "/" + cm_image_name + ".jpg"
    dm_image_path = ProjectPaths.low_energy_images_aligned + "/" + dm_image_name + ".jpg"
    # aligned images
    dm_image = cv2.imread(dm_image_path, cv2.IMREAD_GRAYSCALE)
    cm_image = cv2.imread(cm_image_path, cv2.IMREAD_GRAYSCALE)
    if dm_image is None:
        raise ValueError(f"Error loading DM image with path:\n{dm_image_path}.")
    if cm_image is None:
        raise ValueError(f"Error loading CM image with path:\n{cm_image_path}.")

    # check if fixed image hase the same dimensions as original image
    if fixed_image_is_cm:
        original_image_path = ProjectPaths.subtracted_images + "/" + cm_image_name + ".jpg"
        fixed_image_path = cm_image_path
        fixed_image_name = cm_image_name
    else:
        original_image_path = ProjectPaths.low_energy_images + "/" + dm_image_name + ".jpg"
        fixed_image_path = dm_image_path
        fixed_image_name = dm_image_name

    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    width_org, height_org = original_image.shape
    width_fix , height_fix = cv2.imread(fixed_image_path, cv2.IMREAD_GRAYSCALE).shape
    if width_org != width_fix or height_org != height_fix:
        print(f"Original image dimensions: {width_org}x{height_org}")
        print(f"Fixed image dimensions: {width_fix}x{height_fix}")
        print(f"Original image path: {original_image_path}")
        print(f"Fixed image path: {fixed_image_path}")
        raise ValueError(
            f"Error: fixed image has different dimensions than original image."
        )
    # check if CM and DM images have the same dimensions
    width_cm, height_cm = cm_image.shape
    width_dm, height_dm = dm_image.shape
    if width_cm != width_dm or height_cm != height_dm:
        print(f"CM image dimensions: {width_cm}x{height_cm}")
        print(f"DM image dimensions: {width_dm}x{height_dm}")
        print(f"CM image path: {cm_image_path}")
        print(f"DM image path: {dm_image_path}")
        raise ValueError(
            f"Error: CM and DM images have different dimensions."
        )
    
    filtered_row = annotations[annotations["Image_name"] == cm_image_name]
    if not filtered_row.empty:
        # Extract the label from the filtered row (assuming the label is stored in a column named 'label')
        label = filtered_row.iloc[0]["Pathology Classification/ Follow up"]
    else:
        # If no record is found for the given image_name, raise value error.
        raise ValueError(f"No record found for image name {cm_image_name}")
    label_map = {
        "Normal": -1, # For images having segmentation mask(s) with "normal" label.
        "Benign": 0 ,
        "Malignant": 1,
    }
    label = label_map[label]
    if label == -1:
        print(f"{cm_image_name} has normal({label}). double check it.")
    
    # TODO: create dataset.txt file with the following format:
    # P1_L_CM_MLO.jpg 1 95,125,250,300,0 350,80,450,180,0
    # Column 1: image name.
    # Column 2: Image-level label (1 for Malignant, 0 for Benign).
    # Subsequent Columns: Bounding boxes in min_x,min_y,max_x,max_y,class_id format. Since we only have one object class ("mass"), the class_id is always 0.
    line = f"{fixed_image_name} {label}"
    for bbox in bounding_boxes:
        min_x, max_y = bbox.top_left
        max_x, min_y = bbox.bottom_right
        # for out of frame masses
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > height_cm:
            max_x = height_cm
        if max_y > width_cm:
            max_y = width_cm
        class_id = 0
        line += f" {min_x},{min_y},{max_x},{max_y},{class_id}"
    lines.append(line)

    cv2.imwrite(f"{save_path}{cm_image_name}.jpg", cm_image)
    cv2.imwrite(f"{save_path}{dm_image_name}.jpg", dm_image)

print("number of images: ",len(mass_union_mass_enhancement))
print("number of annotations: ",len(lines))
# write lines to file
with open(f"{save_path}annotations.txt", "w") as f:
    f.write("\n".join(lines))
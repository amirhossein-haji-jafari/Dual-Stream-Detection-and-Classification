
from ..detection_data_preparation.detection_dataset_definitions import mass_union_mass_enhancement
from immutables import ProjectPaths
import numpy as np
import pandas as pd
import cv2
from ..utils.medical_image_utils import (
    # update_images_absolute_paths,
    get_segmented_image_with_mask_indices,
)

def alpha_blend(img, mask):
    """
    Blend two images using an alpha blending technique.

    Parameters:
        - image1 (np.ndarray): The first input image. Should be of shape (height, width, channels).
        - image2 (np.ndarray): The second input image. Should be of the same shape as `image1`.

    Returns:
       - np.ndarray: A blended image of shape (height, width, channels). Channels are typically RGB.

    Notes:
    - Alpha blending combines two images by calculating a weighted sum of their pixel values.
    - The weight for each image is defined by its alpha channel, which typically ranges from 0 to 1.
    - If an image has no alpha channel, it is assumed to have an alpha value of 1 (fully opaque).
    """
    ALPHA = 0.5
    mask *= (ALPHA * mask * 255).astype(np.uint8)
    redImg = np.zeros(img.shape, np.uint8)
    redImg[:, :] = (0, 0, 255)

    redMask = cv2.bitwise_and(
        redImg.astype(np.uint8), redImg.astype(np.uint8), mask=mask.astype(np.uint8)
    )
    cv2.addWeighted(redMask, ALPHA, img, 1, 0, img)
    blended = img.astype(np.uint16) + redMask  # np.expand_dims(mask, axis=-1)
    blended = blended.clip(0, 255)
    return blended.astype(np.uint8)

def main():
    annotations = pd.read_csv(ProjectPaths.annotations_all_sheet_modified)
    segmentations = pd.read_csv(ProjectPaths.segmentations)
    for image_name in mass_union_mass_enhancement:
        cm_image_name = ""
        dm_image_name = ""
        if "_CM_" in image_name:
            cm_image_name = image_name
            dm_image_name = cm_image_name.replace("_CM_","_DM_")
        else:
            dm_image_name = image_name
            cm_image_name = dm_image_name.replace("_DM_", "_CM_")
        filtered_row_cm = annotations[annotations["Image_name"] == cm_image_name]
        filtered_row_dm = annotations[annotations["Image_name"] == dm_image_name]
        if (not filtered_row_cm.empty) and (not filtered_row_dm.empty):
            # Extract the label from the filtered row (assuming the label is stored in a column named 'label')
            label_cm = filtered_row_cm.iloc[0]["Pathology Classification/ Follow up"]
            label_dm = filtered_row_dm.iloc[0]["Pathology Classification/ Follow up"]
            if label_cm != label_dm:
                raise ValueError(
                    f"Labels for {cm_image_name} and {dm_image_name} do not match: {label_cm} != {label_dm}"
                )
        else:
            # If no record is found for the given image_name, raise value error.
            raise ValueError(f"No record found for image name {cm_image_name} or {dm_image_name}")
        
        cm_path = ProjectPaths.subtracted_images + "/" + cm_image_name +".jpg"
        dm_path = ProjectPaths.low_energy_images + "/" + dm_image_name + ".jpg"
        cm_image = cv2.imread(cm_path)
        dm_image = cv2.imread(dm_path)
        if cm_image is None or dm_image is None:
            raise ValueError("Error loading original image.")
        masks_cm = segmentations[segmentations["#filename"] == cm_image_name + ".jpg"][
            "region_shape_attributes"
        ]
        masks_dm = segmentations[segmentations["#filename"] == dm_image_name + ".jpg"][
            "region_shape_attributes"
        ]

       
        # check to see if masks is empty
        if masks_cm.empty:
            print(f"No masks found for image: {cm_image_name} with label {cm_image_name}")
        if masks_dm.empty:
            print(f"No masks found for image: {dm_image_name} with label {dm_image_name}")
        if not masks_cm.empty:
            segmentations_mask_cm = (
                np.array(get_segmented_image_with_mask_indices(cm_image, masks_cm)) > 0
            )
            # Blend, to show the image with segmentation mask
            image_original_blended_with_segmentation_mask_cm = alpha_blend(
                np.array(cm_image), segmentations_mask_cm.astype(int)
            )
            cv2.imwrite(
                filename=f"{ProjectPaths.dual_stream_det}/seg_mask_and_indices_of_masks/{cm_image_name}.jpg",
                img=image_original_blended_with_segmentation_mask_cm,
            )
        if not masks_dm.empty:
            segmentations_mask_dm = (
                np.array(get_segmented_image_with_mask_indices(dm_image, masks_dm)) > 0
            )
            # Blend, to show the image with segmentation mask
            image_original_blended_with_segmentation_mask_dm = alpha_blend(
                np.array(dm_image), segmentations_mask_dm.astype(int)
            )
            cv2.imwrite(
                filename=f"{ProjectPaths.dual_stream_det}/seg_mask_and_indices_of_masks/{dm_image_name}.jpg",
                img=image_original_blended_with_segmentation_mask_dm,
            )

if __name__ == "__main__":
    main()

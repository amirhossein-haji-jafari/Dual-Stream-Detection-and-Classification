"""
dataset utilizes a technique where low-energy mammography images are subtracted from their contrast-enhanced counterparts to generate enhanced images.
This process involves acquiring both low-energy and contrast-enhanced images of the same breast region and angle.
However, due to variations in framing and cropping, the resulting low-energy and subtracted images have slightly different dimensions.
specifically, the images are created from the same object (patient's breast) but differ in size.
we want to register these two images: a low-energy image and its subtracted counterpart.
such that pixels at the same (x, y) location within the breast are mapped to the same corresponding pixel location in both images.
This is a multi-modal image registration problem.
two images of the same object taken with different techniques, resulting in different intensity profiles and slightly different spatial framing.
The goal is to find the geometric transformation that aligns one image (the "moving" image) with the other (the "fixed" image).
the transformation is likely rigid (only translation and rotation) or at most affine (includes scaling and shearing), but not a complex non-rigid warp.
Intensity-Based Registration using Mutual Information (MI):
Instead of assuming a linear relationship between pixel intensities, MI measures how well the intensity distribution of one image can predict the intensity distribution of the other.
"""


import os
from .immutables import ProjectPaths
import SimpleITK as sitk
# import numpy as np
import pandas as pd
import numpy as np
import cv2
# Add tqdm to imports at the top
from tqdm import tqdm

# def register_images_mutual_information(fixed_image_path, moving_image_path):
#     """
#     Registers a moving image to a fixed image using SimpleITK's
#     intensity-based registration with Mattes Mutual Information.
    
#     Args:
#         fixed_image_path (str): Path to the reference image (e.g., the smaller, subtracted image).
#         moving_image_path (str): Path to the image to be aligned (e.g., the larger, low-energy image).
        
#     Returns:
#         numpy.ndarray: The registered (aligned and cropped) moving image as a NumPy array.
#     """
#     MAX_THREADS = 24
#     sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(MAX_THREADS)

#     # 1. Load images into SimpleITK format
#     # Read images as 2D and cast them to a floating point type for the registration process
#     fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
#     moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

#     # 2. Set up the registration framework
#     registration_method = sitk.ImageRegistrationMethod()

#     # 3. Define the transformation type
#     # Euler2DTransform covers translation and rotation.
#     # Use Similarity2DTransform for rotation + translation + isotropic scaling.
#     # initial_transform = sitk.Euler2DTransform()
#     initial_transform = sitk.Similarity2DTransform()

#     initial_transform = sitk.CenteredTransformInitializer(
#     fixed_image,
#     moving_image,
#     initial_transform,
#     sitk.CenteredTransformInitializerFilter.GEOMETRY
#     )

#     registration_method.SetInitialTransform(initial_transform, inPlace=False)

#     # 4. Set the similarity metric: Mattes Mutual Information
#     # This is the key for multi-modal registration.
#     # numberOfHistogramBins determines the precision of the MI calculation.
#     registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=200)
#     registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
#     registration_method.SetMetricSamplingPercentage(0.9) # Use 30% of pixels for speed

#     # 5. Set the optimizer
#     # Adjust learningRate and numberOfIterations for your data.
#     registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, 
#                                                       numberOfIterations=2000, 
#                                                       estimateLearningRate=registration_method.EachIteration)
#     registration_method.SetOptimizerScalesFromPhysicalShift()

#     # 6. Set the interpolator
#     registration_method.SetInterpolator(sitk.sitkLinear)

#     # 7. Execute the registration
#     # print("Starting registration...")
#     final_transform = registration_method.Execute(fixed_image, moving_image)
#     # print("Registration finished.")
    
#     # --- Post-Execution Analysis ---
#     # print(f"Final Transform: {final_transform}")
#     # print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
#     # print(f"Final metric value: {registration_method.GetMetricValue()}")

#     # 8. Resample the moving image to align with the fixed image
#     # The Resample function applies the found transform and automatically
#     # crops the output to the dimensions of the 'fixed_image'.
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed_image) # Use fixed image grid as reference
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(0) # Pixels outside the moving image are set to 0
#     resampler.SetTransform(final_transform)

#     registered_image_sitk = resampler.Execute(moving_image)
    
#     # Convert back to NumPy array for use with OpenCV or Matplotlib
#     registered_image_np = sitk.GetArrayFromImage(registered_image_sitk)
    
#     return registered_image_np


# def register_images_mutual_information(fixed_image_path, moving_image_path):
#     """
#     Registers a moving image to a fixed image using SimpleITK's
#     intensity-based registration with Mattes Mutual Information.
#     ...
#     """
#     MAX_THREADS = 24
#     sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(MAX_THREADS)

#     # 1. Load images into SimpleITK format (preserve original spacing if present)
#     fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
#     moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

#     # --- PREPROCESSING: Histogram matching + simple denoise ---
#     matcher = sitk.HistogramMatchingImageFilter()
#     matcher.SetNumberOfHistogramLevels(128)
#     matcher.SetNumberOfMatchPoints(7)
#     matcher.ThresholdAtMeanIntensityOn()
#     moving_matched = matcher.Execute(moving_image, fixed_image)

#     # Optional denoising (bilateral or curvature flow)
#     fixed_smooth = sitk.CurvatureFlow(image1=fixed_image, timeStep=0.125, numberOfIterations=5)
#     moving_smooth = sitk.CurvatureFlow(image1=moving_matched, timeStep=0.125, numberOfIterations=5)

#     # --- CREATE BREAST MASKS (Otsu + keep largest CC) and combine masks ---
#     def largest_cc_mask(img):
#         mask = sitk.OtsuThreshold(img, 0, 1)
#         mask = sitk.BinaryFillhole(mask)
#         cc = sitk.ConnectedComponent(mask)
#         stats = sitk.LabelShapeStatisticsImageFilter()
#         stats.Execute(cc)
#         if stats.GetNumberOfLabels() == 0:
#             return mask
#         # find largest label
#         largest_label = max(stats.GetLabels(), key=lambda L: stats.GetPhysicalSize(L))
#         largest = sitk.Equal(cc, largest_label)
#         return sitk.Cast(largest, sitk.sitkUInt8)

#     # fixed_mask = largest_cc_mask(fixed_smooth)
#     # moving_mask = largest_cc_mask(moving_smooth)
#     # intersection to get robust common region
#     # metric_mask = sitk.Cast(sitk.And(fixed_mask, moving_mask), sitk.sitkUInt8)

#     # 2. Set up the registration framework
#     registration_method = sitk.ImageRegistrationMethod()

#     # 3. Initial transform via geometry-based initializer (centered)
#     # initial_transform = sitk.Euler2DTransform()
#     initial_transform = sitk.Similarity2DTransform()
#     initial_transform = sitk.CenteredTransformInitializer(fixed_smooth,
#                                                           moving_smooth,
#                                                           initial_transform,
#                                                           sitk.CenteredTransformInitializerFilter.GEOMETRY)
#     registration_method.SetInitialTransform(initial_transform, inPlace=False)

#     # 4. Metric: Mattes Mutual Information with mask
#     registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
#     registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)   # reproducible & stable
#     registration_method.SetMetricSamplingPercentage(0.85)  # smaller % at full resolution

#     # pass mask to metric to ignore background/labels
#     # registration_method.SetMetricFixedMask(metric_mask)
#     # registration_method.SetMetricMovingMask(metric_mask)

#     # 5. Multi-resolution pyramid (recommended)
#     registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
#     registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
#     registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

#     # 6. Optimizer: Regular step gradient descent (robust)
#     registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
#                                                                  minStep=1e-4,
#                                                                  numberOfIterations=1000,
#                                                                  relaxationFactor=0.5)
#     registration_method.SetOptimizerScalesFromPhysicalShift()

#     # 7. Interpolator
#     registration_method.SetInterpolator(sitk.sitkLinear)

#     # 8. Execute
#     final_transform = registration_method.Execute(fixed_smooth, moving_smooth)

#     # Debug prints (you can save these to logs)
#     # print("Optimizer stop condition:", registration_method.GetOptimizerStopConditionDescription())
#     # print("Final metric value:", registration_method.GetMetricValue())

#     # 9. Resample moving image (use original moving_image spacing/origin/direction)
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed_image)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(0)
#     resampler.SetTransform(final_transform)
#     registered_image_sitk = resampler.Execute(moving_image)

#     registered_image_np = sitk.GetArrayFromImage(registered_image_sitk)
#     return registered_image_np

# def register_images_mutual_information(fixed_image_path, moving_image_path):
#     """
#     Registers a moving image to a fixed image using SimpleITK's
#     intensity-based registration with Mattes Mutual Information.
#     ...
#     """
#     MAX_THREADS = 24
#     sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(MAX_THREADS)

#     # 1. Load images into SimpleITK format (preserve original spacing if present)
#     fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
#     moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

#     # --- PREPROCESSING: Histogram matching + simple denoise ---
#     matcher = sitk.HistogramMatchingImageFilter()
#     matcher.SetNumberOfHistogramLevels(128)
#     matcher.SetNumberOfMatchPoints(7)
#     matcher.ThresholdAtMeanIntensityOn()
#     moving_matched = matcher.Execute(moving_image, fixed_image)

#     # Optional denoising (bilateral or curvature flow)
#     fixed_smooth = sitk.CurvatureFlow(image1=fixed_image, timeStep=0.125, numberOfIterations=5)
#     moving_smooth = sitk.CurvatureFlow(image1=moving_matched, timeStep=0.125, numberOfIterations=5)

#     # --- CREATE BREAST MASKS (Otsu + keep largest CC) and combine masks ---
#     def largest_cc_mask(img):
#         mask = sitk.OtsuThreshold(img, 0, 1)
#         mask = sitk.BinaryFillhole(mask)
#         cc = sitk.ConnectedComponent(mask)
#         stats = sitk.LabelShapeStatisticsImageFilter()
#         stats.Execute(cc)
#         if stats.GetNumberOfLabels() == 0:
#             return mask
#         # find largest label
#         largest_label = max(stats.GetLabels(), key=lambda L: stats.GetPhysicalSize(L))
#         largest = sitk.Equal(cc, largest_label)
#         return sitk.Cast(largest, sitk.sitkUInt8)

#     # fixed_mask = largest_cc_mask(fixed_smooth)
#     # moving_mask = largest_cc_mask(moving_smooth)
#     # intersection to get robust common region
#     # metric_mask = sitk.Cast(sitk.And(fixed_mask, moving_mask), sitk.sitkUInt8)

#     fixed_norm = sitk.Normalize(fixed_smooth)
#     moving_norm = sitk.Normalize(moving_smooth)

#     # 2. Set up the registration framework
#     registration_method = sitk.ImageRegistrationMethod()

#     # 3. Initial transform via geometry-based initializer (centered)
#     # initial_transform = sitk.Euler2DTransform()
#     initial_transform = sitk.Similarity2DTransform()
#     initial_transform = sitk.CenteredTransformInitializer(fixed_norm,
#                                                           moving_norm,
#                                                           initial_transform,
#                                                           sitk.CenteredTransformInitializerFilter.GEOMETRY)
#     registration_method.SetInitialTransform(initial_transform, inPlace=False)

#     # 4. Metric: Mattes Mutual Information with mask
#     registration_method.SetMetricAsCorrelation()
#     registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)   # reproducible & stable
#     registration_method.SetMetricSamplingPercentage(0.85)  # smaller % at full resolution

#     # pass mask to metric to ignore background/labels
#     # registration_method.SetMetricFixedMask(metric_mask)
#     # registration_method.SetMetricMovingMask(metric_mask)

#     # 5. Multi-resolution pyramid (recommended)
#     registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
#     registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2,1,0])
#     registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

#     # 6. Optimizer: Regular step gradient descent (robust)
#     registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
#                                                                  minStep=1e-4,
#                                                                  numberOfIterations=1000,
#                                                                  relaxationFactor=0.5)
#     registration_method.SetOptimizerScalesFromPhysicalShift()

#     # 7. Interpolator
#     registration_method.SetInterpolator(sitk.sitkLinear)

#     # 8. Execute
#     final_transform = registration_method.Execute(fixed_smooth, moving_smooth)

#     # Debug prints (you can save these to logs)
#     # print("Optimizer stop condition:", registration_method.GetOptimizerStopConditionDescription())
#     # print("Final metric value:", registration_method.GetMetricValue())

#     # 9. Resample moving image (use original moving_image spacing/origin/direction)
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed_image)
#     resampler.SetInterpolator(sitk.sitkLinear)
#     resampler.SetDefaultPixelValue(0)
#     resampler.SetTransform(final_transform)
#     registered_image_sitk = resampler.Execute(moving_image)

#     registered_image_np = sitk.GetArrayFromImage(registered_image_sitk)
#     return registered_image_np


def register_images_mutual_information(fixed_image_path, moving_image_path):
    """
    Registers a moving image to a fixed image using gradient magnitude images
    and mean squares metric for registration.
    """
    MAX_THREADS = 24
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(MAX_THREADS)

    # 1. Load images into SimpleITK format
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # --- PREPROCESSING: Compute gradient magnitude images ---
    grad_filter = sitk.GradientMagnitudeImageFilter()
    
    # Optional: Smooth before gradient to reduce noise sensitivity
    gaussian = sitk.DiscreteGaussianImageFilter()
    gaussian.SetVariance(1.0)
    
    fixed_smooth = gaussian.Execute(fixed_image)
    moving_smooth = gaussian.Execute(moving_image)
    
    # Compute gradient magnitude images
    fixed_grad = grad_filter.Execute(fixed_smooth)
    moving_grad = grad_filter.Execute(moving_smooth)
    
    # Normalize gradient images to [0,1] range
    fixed_grad = sitk.Normalize(fixed_grad)
    moving_grad = sitk.Normalize(moving_grad)

    # 2. Set up the registration framework
    registration_method = sitk.ImageRegistrationMethod()

    # 3. Initial transform via geometry-based initializer (centered)
    initial_transform = sitk.Similarity2DTransform()
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_grad,
        moving_grad,
        initial_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 4. Metric: Mean Squares on gradient magnitude images
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.85)

    # 5. Multi-resolution pyramid (recommended)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 6. Optimizer: Regular step gradient descent with adjusted parameters
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,  # Might need adjustment for gradient images
        minStep=1e-4,
        numberOfIterations=1000,
        relaxationFactor=0.5
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # 7. Interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 8. Execute registration on gradient magnitude images
    final_transform = registration_method.Execute(fixed_grad, moving_grad)

    # 9. Apply transform to original moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)
    registered_image_sitk = resampler.Execute(moving_image)

    registered_image_np = sitk.GetArrayFromImage(registered_image_sitk)
    return registered_image_np

def main():
    """
    Main function to orchestrate the image registration process.

    This function iterates through a CSV file (annotations_all_sheet_modified)
    containing information about pairs of mammography images (contrast-enhanced
    and low-energy). It performs the following steps for each image pair:

    1.  Loads image data from the specified paths.
    2.  Determines which image is the "fixed" (reference) and which is the
        "moving" image based on their dimensions.
    3.  Calls the `register_images_mutual_information` function to perform
        the registration using SimpleITK's mutual information method.
    4.  Saves the registered image to the appropriate output directory.

    Args:
        None

    Returns:
        None.  The function performs actions (saving images) rather than
        returning a value.

    Raises:
        ValueError: If an image cannot be loaded from its specified path.

    """
    annotations_all_sheet_modified = pd.read_csv(ProjectPaths.annotations_all_sheet_modified)
    registered_images_names = []
    
    # Filter out CM images first to get accurate total
    dm_rows = annotations_all_sheet_modified[~annotations_all_sheet_modified["Image_name"].str.contains("_CM_")]
    
    # Create progress bar
    pbar = tqdm(total=len(dm_rows), desc="Registering images", unit="image")
    
    for index, row in dm_rows.iterrows():
        image_name = row["Image_name"]
        dm_image_name = image_name.strip()
        cm_image_name = dm_image_name.replace("_DM_","_CM_").strip()
        dm_path = row["Absolute_path"]
        cm_path = ProjectPaths.subtracted_images + "/" + cm_image_name + ".jpg"
        
        # Update progress bar description
        pbar.set_description(f"Processing {dm_image_name}")
        
        try:
            dm_image = cv2.imread(dm_path, cv2.IMREAD_GRAYSCALE)
            cm_image = cv2.imread(cm_path, cv2.IMREAD_GRAYSCALE)
            
            if dm_image is None:
                raise ValueError(f"Error loading DM image with path:\n{dm_path}.")
            if cm_image is None:
                raise ValueError(f"Error loading CM image with path:\n{cm_path}.")
                
            dm_image_h, dm_image_w = dm_image.shape
            cm_image_h, cm_image_w = cm_image.shape
            
            # compare perimeter of images
            is_fixed_image_dm = (dm_image_h + dm_image_w) <= (cm_image_h + cm_image_w)
            
            if is_fixed_image_dm:
                FIXED_IMAGE_PATH = dm_path
                MOVING_IMAGE_PATH = cm_path
                fixed_img_orig = dm_image
                moving_img_orig = cm_image
            else:
                FIXED_IMAGE_PATH = cm_path
                MOVING_IMAGE_PATH = dm_path
                fixed_img_orig = cm_image
                moving_img_orig = dm_image
            
            # Perform registration
            registered_img = register_images_mutual_information(FIXED_IMAGE_PATH, MOVING_IMAGE_PATH)
            
            if is_fixed_image_dm:
                cv2.imwrite(ProjectPaths.low_energy_images_aligned + "/" + dm_image_name + ".jpg", fixed_img_orig)
                cv2.imwrite(ProjectPaths.subtracted_images_aligned + "/" + cm_image_name + ".jpg", registered_img)
                registered_images_names.append(cm_image_name)
            else:
                cv2.imwrite(ProjectPaths.subtracted_images_aligned + "/" + cm_image_name + ".jpg", fixed_img_orig)
                cv2.imwrite(ProjectPaths.low_energy_images_aligned + "/" + dm_image_name + ".jpg", registered_img)
                registered_images_names.append(dm_image_name)
                
        except Exception as e:
            print(f"\nError processing {dm_image_name}: {str(e)}")
            
        # Update progress bar
        pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    print("\nRegistration complete!")
    print("Names of registered (aligned) images:", registered_images_names, sep="\n")


def create_checkerboard_visualization(fixed_img, moving_img, patch_size=50):
    """
    Creates a checkerboard visualization from two images of the same size.
    """
    h, w = fixed_img.shape
    checkerboard = np.zeros((h, w), dtype=fixed_img.dtype)
    
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            # Determine which image to use based on the patch's position
            use_fixed = ((i // patch_size) % 2 == 0 and (j // patch_size) % 2 == 0) or \
                        ((i // patch_size) % 2 != 0 and (j // patch_size) % 2 != 0)

            h_slice = slice(i, min(i + patch_size, h))
            w_slice = slice(j, min(j + patch_size, w))

            if use_fixed:
                checkerboard[h_slice, w_slice] = fixed_img[h_slice, w_slice]
            else:
                checkerboard[h_slice, w_slice] = moving_img[h_slice, w_slice]
                
    return checkerboard

def crop_center(img, target_h, target_w):
    """
    Crops the center of an image to a target height and width.
    """
    h, w = img.shape[:2]
    start_w = w // 2 - target_w // 2
    start_h = h // 2 - target_h // 2
    return img[start_h:start_h + target_h, start_w:start_w + target_w]

def visualize_registration_for_case(fixed_path, moving_path, output_dir):
    """
    Performs registration for a single case and saves before/after visualizations.
    """
    print(f"Visualizing registration for:\n  Fixed: {os.path.basename(fixed_path)}\n  Moving: {os.path.basename(moving_path)}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load original images
    fixed_img_orig = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)
    moving_img_orig = cv2.imread(moving_path, cv2.IMREAD_GRAYSCALE)
    
    if fixed_img_orig is None or moving_img_orig is None:
        print("Error: Could not load one or both images.")
        return

    # --- "BEFORE" VISUALIZATION ---
    # To compare, we need to make them the same size. We'll crop the center
    # of the larger (moving) image to match the smaller (fixed) image.
    h_fixed, w_fixed = fixed_img_orig.shape
    moving_img_cropped = crop_center(moving_img_orig, h_fixed, w_fixed)
    
    # Create and save "before" visualizations
    checkerboard_before = create_checkerboard_visualization(fixed_img_orig, moving_img_cropped)
    
    before_overlay_path = os.path.join(output_dir, "overlay_before.png")
    before_checker_path = os.path.join(output_dir, "checkerboard_before.png")
    cv2.imwrite(before_checker_path, checkerboard_before)
    print(f"Saved 'before' visualizations to:\n  {before_overlay_path}\n  {before_checker_path}")


    # --- PERFORM REGISTRATION ---
    registered_moving_img = register_images_mutual_information(fixed_path, moving_path)
    
    # Convert to 8-bit for visualization
    registered_moving_img_8bit = cv2.normalize(registered_moving_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # --- "AFTER" VISUALIZATION ---
    # The registered image is already aligned and cropped to the fixed image's size.
    checkerboard_after = create_checkerboard_visualization(fixed_img_orig, registered_moving_img_8bit)
    
    after_overlay_path = os.path.join(output_dir, "overlay_after.png")
    after_checker_path = os.path.join(output_dir, "checkerboard_after.png")
    cv2.imwrite(after_checker_path, checkerboard_after)
    print(f"Saved 'after' visualizations to:\n  {after_overlay_path}\n  {after_checker_path}")
    print("\nVisualization complete.")


if __name__ == "__main__":
    main()

    # Fixed image (smaller one)
    FIXED_IMAGE_PATH = os.path.join(ProjectPaths.subtracted_images, "P66_L_CM_CC.jpg")
    # Moving image (larger one)
    MOVING_IMAGE_PATH = os.path.join(ProjectPaths.low_energy_images, "P66_L_DM_CC.jpg")
    
    # Define where to save the output images
    OUTPUT_VISUALIZATION_DIR = os.path.join(ProjectPaths.dual_stream_det, "registration_visualizations", "P66_L_CC")

    # visualize_registration_for_case(FIXED_IMAGE_PATH, MOVING_IMAGE_PATH, OUTPUT_VISUALIZATION_DIR)
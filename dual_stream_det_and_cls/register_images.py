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
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# Optimization Monitoring
def command_iteration(method):
    if method.GetOptimizerIteration() == 0:
        print("Starting registration...")
    if method.GetOptimizerIteration() % 50 == 0:
        print(f"Iteration: {method.GetOptimizerIteration()}")
        print(f"Metric value: {method.GetMetricValue()}")


def register_images(fixed_image_path,
                                       moving_image_path,
                                       transform_type='euler2d',
                                       optimizer_type='gd',
                                       metric='mattes',
                                       use_gradient=False,
                                       histogram_match=False,
                                       num_threads=24,
                                       sampling_percentage=0.90,
                                       shrink_factors=[8, 4, 2, 1] ,
                                       smoothing_sigmas=[4, 2, 1, 0]):
    """
    Registers a moving image to a fixed image using SimpleITK.

    Args:
        fixed_image_path (str): Path to reference image.
        moving_image_path (str): Path to moving image.
        transform_type (str): 'euler2d' or 'similarity2d'.
        optimizer_type (str): 'regular' for RegularStepGradientDescent or 'gradient' for GradientDescent.
        metric (str): 'mattes', 'mean_squares', or 'correlation'.
            mattes: Intensity-based registration with Mattes Mutual Information.
        use_gradient (bool): If True compute gradient magnitude images and register those.
        histogram_match (bool): If True perform histogram matching (when not using gradients).
        num_threads (int): Number of threads for SimpleITK.
        sampling_percentage (float): Metric sampling percentage.
        shrink_factors (list): Multi-resolution shrink factors.
        smoothing_sigmas (list): Multi-resolution smoothing sigmas.

    Returns:
        numpy.ndarray: Registered moving image resampled to fixed image grid as a NumPy array.
    """
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(num_threads)

    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Preprocessing
    if use_gradient:
        # smooth then compute gradient magnitude
        gaussian = sitk.DiscreteGaussianImageFilter()
        gaussian.SetVariance(1.0)
        fixed_smooth = gaussian.Execute(fixed_image)
        moving_smooth = gaussian.Execute(moving_image)
        grad_filter = sitk.GradientMagnitudeImageFilter()
        fixed_for_reg = sitk.Normalize(grad_filter.Execute(fixed_smooth))
        moving_for_reg = sitk.Normalize(grad_filter.Execute(moving_smooth))
    else:
        fixed_for_reg = fixed_image
        if histogram_match:
            matcher = sitk.HistogramMatchingImageFilter()
            matcher.SetNumberOfHistogramLevels(128)
            matcher.SetNumberOfMatchPoints(7)
            matcher.ThresholdAtMeanIntensityOn()
            moving_matched = matcher.Execute(moving_image, fixed_image)
        else:
            moving_matched = moving_image

        # optional denoising
        fixed_smooth = sitk.CurvatureFlow(image1=fixed_for_reg, timeStep=0.125, numberOfIterations=5)
        moving_smooth = sitk.CurvatureFlow(image1=moving_matched, timeStep=0.125, numberOfIterations=5)
        fixed_for_reg = sitk.Normalize(fixed_smooth)
        moving_for_reg = sitk.Normalize(moving_smooth)

    # Registration method setup
    registration_method = sitk.ImageRegistrationMethod()
    # Metric selection
    metric_lower = metric.lower()
    if metric_lower == 'mattes':
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=256)
    elif metric_lower == 'mean_squares' or metric_lower == 'meansquares':
        registration_method.SetMetricAsMeanSquares()
    elif metric_lower == 'correlation':
        registration_method.SetMetricAsCorrelation()
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Initial transform
    if transform_type.lower() == 'euler2d':
        base_transform = sitk.Euler2DTransform()
    else:
        base_transform = sitk.Similarity2DTransform()
    # registration_method.SetInitialTransform(base_transform, inPlace=False)
    # Trying both GEOMETRY and MOMENTS initialization
    initial_transform_geo = sitk.CenteredTransformInitializer(
        fixed_for_reg,
        moving_for_reg,
        base_transform,
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    # Set this transform on the registration method to evaluate it
    registration_method.SetInitialTransform(initial_transform_geo)
    # metric_geometry = registration_method.MetricEvaluate(fixed_for_reg, moving_for_reg)
    # initial_transform_moments = sitk.CenteredTransformInitializer(
    #     fixed_for_reg, 
    #     moving_for_reg,
    #     base_transform,
    #     sitk.CenteredTransformInitializerFilter.MOMENTS
    # )
    # # Set this transform on the registration method to evaluate it
    # registration_method.SetInitialTransform(initial_transform_moments)
    # metric_moments = registration_method.MetricEvaluate(fixed_for_reg, moving_for_reg)
    # # Choose better initialization
    # if metric_moments < metric_geometry:
    #     initial_transform = initial_transform_moments
    # else:
    #     initial_transform = initial_transform_geo
    # registration_method.SetInitialTransform(initial_transform, inPlace=False)

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(sampling_percentage)

    # Multi-resolution
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Optimizer selection
    if optimizer_type.lower() in ('regular', 'regularstep', 'regularstepgradientdescent', 'rsgd', 'stepgradient' ):
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=2.0,
            minStep=1e-6,
            numberOfIterations=2000,
            relaxationFactor=0.7,
            gradientMagnitudeTolerance=1e-6
        )
    elif optimizer_type.lower() in ('gradient', 'gradientdescent', 'gd'):
        # Basic gradient descent; parameters can be tuned by caller
        registration_method.SetOptimizerAsGradientDescent(learningRate=0.1,
                                                          numberOfIterations=2000,
                                                          convergenceMinimumValue=1e-6,
                                                          convergenceWindowSize=10,
                                                          estimateLearningRate=registration_method.EachIteration
                                                          )
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")\
        
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # registration_method.SetOptimizerScalesFromIndexShift()
    registration_method.SetInterpolator(sitk.sitkLinear)

    # Execute registration on chosen images
    final_transform = registration_method.Execute(fixed_for_reg, moving_for_reg)

    # Log info
    try:
        with open(ProjectPaths.registration_logs, "a") as log_file:
            log_file.write(f"Registration between fixed {os.path.basename(fixed_image_path)} and moving {os.path.basename(moving_image_path)}\n")
            log_file.write("Optimizer stop condition: " + registration_method.GetOptimizerStopConditionDescription() + "\n")
            log_file.write("Final metric value: " + str(registration_method.GetMetricValue()) + "\n\n")
    except Exception:
        pass

    # Resample moving image to fixed image grid (use original moving_image for appearance)
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
    3.  Calls the `register_images` function to perform
        the registration using SimpleITK.
    4.  Saves the registered image to the appropriate output directory.

    Args:
        None

    Returns:
        None.  The function performs actions (saving images) rather than
        returning a value.

    Raises:
        ValueError: If an image cannot be loaded from its specified path.

    """
    # remove previously logged registration info
    if os.path.exists(ProjectPaths.registration_logs):
        os.remove(ProjectPaths.registration_logs)

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
            registered_img = register_images(FIXED_IMAGE_PATH, MOVING_IMAGE_PATH)
            
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
    registered_moving_img = register_images(fixed_path, moving_path)
    
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
    OUTPUT_VISUALIZATION_DIR = os.path.join(ProjectPaths.visualize_registration, "P66_L_CC")

    # visualize_registration_for_case(FIXED_IMAGE_PATH, MOVING_IMAGE_PATH, OUTPUT_VISUALIZATION_DIR)
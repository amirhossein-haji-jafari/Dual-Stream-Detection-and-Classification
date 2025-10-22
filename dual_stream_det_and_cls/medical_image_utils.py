import json
from collections import namedtuple
from .immutables import ProjectPaths
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import os
import logging


def alpha_blend(image, mask, color_bgr=(0, 0, 255), alpha=0.4):
    """
    Overlays a colored mask on an image with a specified transparency.
    Args:
        image (np.ndarray): The background image (BGR). Should be of shape (height, width, channels).
        mask (np.ndarray): The boolean or binary (0/255) mask.
        color_bgr (tuple): The BGR color for the overlay.
        alpha (float): The transparency of the overlay.
    Returns:
        np.ndarray: The blended image.
        
    Notes:
    - Alpha blending combines two images by calculating a weighted sum of their pixel values.
    - The weight for each image is defined by its alpha channel, which typically ranges from 0 to 1.
    - If an image has no alpha channel, it is assumed to have an alpha value of 1 (fully opaque).
    """
    output = image.copy()
    # Ensure mask is a single-channel boolean array
    if mask.ndim == 3:
        # If mask has channels, collapse it to a boolean mask
        mask = mask[:, :, 0] > 0
    else:
        mask = mask > 0

    # The region of interest (ROI) from the original image where the mask is True
    roi = output[mask]

    # Create a solid color overlay for just the ROI
    color_overlay_roi = np.full(roi.shape, color_bgr, dtype=roi.dtype)

    # Blend only the ROI with the color overlay
    blended_roi = cv2.addWeighted(color_overlay_roi, alpha, roi, 1 - alpha, 0)

    # Place the blended ROI back into the output image
    output[mask] = blended_roi
    
    return output


def get_polygon_formatted(x_points, y_points):
    """
    Convert lists of x and y points into a list of tuples representing polygon vertices.

    Parameters:
    - x_points (list): A list of x-coordinates for the polygon vertices.
    - y_points (list): A list of y-coordinates for the polygon vertices.

    Returns:
    - list: A list of tuples, each representing a vertex of the polygon in the format (x, y).

    Notes:
    - The function assumes that both `x_points` and `y_points` are lists of equal length.
    - If the lengths differ or if one of the lists is empty, an error will be raised during execution.

    Examples:
    >>> get_polygon_formatted([0, 1, 2], [0, 1, 0])
    [(0, 0), (1, 1), (2, 0)]

    >>> get_polygon_formatted([-1, 0, 1], [-1, 0, 1])
    [(-1, -1), (0, 0), (1, -1)]
    """
    points = []
    for i in range(len(x_points)):
        points.append((x_points[i], y_points[i]))
    return points


def get_segmented_image(image, masks):
    """
    create segmented image according to a patient's breast image and corresponding segmentaion mask.

    Parameters:
        - image (int): patient's breast image.
        - masks (list): A list of string representations of segmentation masks.
            Each mask is expected to be in JSON format with at least one key called 'name'.
            'name' can be one of the following types:
                - "polygon"
                - "circle"
                - "point"
                - "ellipse"

    Returns:
        - img_mask: A binary image with 1s representing segmentation mask.

    Notes:
    - This function supports masks of types 'point', 'circle', 'ellipse', and 'polygon'.
    - If a mask is a string, it is expected to be in JSON format with 'name' and optional attributes.
    - The function assumes that the input data is well-formed according to the expected structure.
    """
    img_mask = Image.new("L", (image.shape[1], image.shape[0]), 0)
    for mask in masks:
        if mask == "{}":
            continue
        mask = json.loads(mask)
        if mask["name"] == "polygon":
            poly = get_polygon_formatted(mask["all_points_x"], mask["all_points_y"])
            ImageDraw.Draw(img_mask).polygon(poly, outline=1, fill=1)
        elif (
            mask["name"] == "ellipse"
            or mask["name"] == "circle"
            or mask["name"] == "point"
        ):
            if mask["name"] == "circle":
                mask["rx"] = mask["ry"] = mask["r"]
            elif mask["name"] == "point":
                mask["rx"] = mask["ry"] = 25
            ellipse = [
                (mask["cx"] - mask["rx"], mask["cy"] - mask["ry"]),
                (mask["cx"] + mask["rx"], mask["cy"] + mask["ry"]),
            ]
            ImageDraw.Draw(img_mask).ellipse(ellipse, outline=1, fill=1)
    return img_mask

def get_segmented_image_with_mask_indices(image, masks):
    """
    create segmented image according to a patient's breast image and corresponding segmentaion mask and each mask's index.

    Parameters:
        - image (int): patient's breast image.
        - masks (list): A list of string representations of segmentation masks.
            Each mask is expected to be in JSON format with at least one key called 'name'.
            'name' can be one of the following types:
                - "polygon"
                - "circle"
                - "point"
                - "ellipse"

    Returns:
        - img_mask: A binary image with 1s representing segmentation mask.

    Notes:
    - This function supports masks of types 'point', 'circle', 'ellipse', and 'polygon'.
    - If a mask is a string, it is expected to be in JSON format with 'name' and optional attributes.
    - The function assumes that the input data is well-formed according to the expected structure.
    """
    img_mask = Image.new("L", (image.shape[1], image.shape[0]), 0)
    masks_indices = masks.index.values
    counter = 0
    indices_font = ImageFont.truetype('/home/monstalinux/final-project/evaluate_datasets/CaskaydiaCoveNerdFontMono-Regular.ttf', 32)
    for mask in masks:
        if mask == "{}":
            continue
        mask = json.loads(mask)
        if mask["name"] == "polygon":
            poly = get_polygon_formatted(mask["all_points_x"], mask["all_points_y"])
            ImageDraw.Draw(img_mask).polygon(poly, outline=1, fill=None, width=4)
            for i in range(len(poly)):
                ImageDraw.Draw(img_mask).text((poly[i]), f"{masks_indices[counter]}", font=indices_font ,fill=1)
            counter += 1
        elif (
            mask["name"] == "ellipse"
            or mask["name"] == "circle"
            or mask["name"] == "point"
        ):
            if mask["name"] == "circle":
                mask["rx"] = mask["ry"] = mask["r"]
            elif mask["name"] == "point":
                mask["rx"] = mask["ry"] = 25
            ellipse = [
                (mask["cx"] - mask["rx"], mask["cy"] - mask["ry"]),
                (mask["cx"] + mask["rx"], mask["cy"] + mask["ry"]),
            ]
            ImageDraw.Draw(img_mask).ellipse(ellipse, outline=1, fill=None, width=4)
            ImageDraw.Draw(img_mask).text((mask["cx"],mask["cy"]), f"{masks_indices[counter]}", font=indices_font ,fill=1)
            counter += 1

        ImageDraw.Draw(img_mask).text((0,0), f"#masks({len(masks_indices)})", font=indices_font ,fill=1)
    return img_mask

def get_bounding_boxes(masks):
    """
    Calculate bounding boxes for different types of masks.

    Parameters:
        - masks (list): A list of string representations of segmentation masks.
        Each mask is expected to be in JSON format with al least one key called 'name'.
        'name' can be one of the following types:
            - "polygon"
            - "circle"
            - "point"
            - "ellipse"

    Returns:
       - list: A list of bounding boxes. Each bounding box is a list of four named tuples, representing the coordinates
       of the bottom_left, top_left, top_right, and bottom_right corners of the bounding box.

    Notes:
    - This function supports masks of types 'point', 'circle', 'ellipse', and 'polygon'.
    - If a mask is a string, it is expected to be in JSON format with 'name' and optional attributes.
    - The function assumes that the input data is well-formed according to the expected structure.
    """
    Bounding_Box = namedtuple(
        "Bounding_box", ["bottom_left", "top_left", "top_right", "bottom_right"]
    )
    bounding_boxes = []
    for mask in masks:
        if mask == "{}":
            continue
        mask = json.loads(mask)
        if mask["name"] == "polygon":
            points = get_polygon_formatted(mask["all_points_x"], mask["all_points_y"])
            x_coordinates, y_coordinates = zip(*points)
            min_x, min_y, max_x, max_y = (
                min(x_coordinates),
                min(y_coordinates),
                max(x_coordinates),
                max(y_coordinates),
            )
            BBox = Bounding_Box(
                (min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)
            )
            # bounding_box = [
            #     (min_x, min_y),
            #     (min_x, max_y),
            #     (max_x, max_y),
            #     (max_x, min_y),
            # ]
            # bounding_boxes.append(bounding_box)
            bounding_boxes.append(BBox)
        elif (
            mask["name"] == "ellipse"
            or mask["name"] == "circle"
            or mask["name"] == "point"
        ):
            if mask["name"] == "circle":
                mask["rx"] = mask["ry"] = mask["r"]
            elif mask["name"] == "point":
                mask["rx"] = mask["ry"] = 25
            points = [
                (mask["cx"] - mask["rx"], mask["cy"] - mask["ry"]),
                (mask["cx"] + mask["rx"], mask["cy"] + mask["ry"]),
            ]
            x_coordinates, y_coordinates = zip(*points)
            min_x, min_y, max_x, max_y = (
                min(x_coordinates),
                min(y_coordinates),
                max(x_coordinates),
                max(y_coordinates),
            )
            BBox = Bounding_Box(
                (min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)
            )
            # bounding_box = [
            #     (min_x, min_y),
            #     (min_x, max_y),
            #     (max_x, max_y),
            #     (max_x, min_y),
            # ]
            # bounding_boxes.append(bounding_box)
            bounding_boxes.append(BBox)
    return bounding_boxes


def get_segmented_image_with_bounding_boxes(image, bounding_boxes):
    """
    Draw bounding boxes on an image.
    Generates an image where each bounding box in `bounding_boxes` is filled with white pixels on a black background.

    Parameters:
        - image (numpy.ndarray): Input image array representing the patient's breast image.
        - bounding_boxes (list): A list of bounding boxes. Each bounding box is a list of four tuples, representing the coordinates
        of the bottom-left, top-left, top-right, and bottom-right corners of the bounding box.

    Returns:
        - numpy.ndarray: Image with bounding boxes drawn on it.

    Notes:
    - Each bounding box in the `bounding_boxes` list should contain four integers representing the coordinates
      of a rectangular region.
    """
    img_mask = Image.new("L", (image.shape[1], image.shape[0]), 0)
    for bbox in bounding_boxes:
        ImageDraw.Draw(img_mask).polygon(bbox, outline=1, fill=1)

    return img_mask


def show(
    image,
    title="Image",
    max_height=720,
    dpi=100,
    show_in_separate_window=False,
    show_in_line=False,
    save=True,
    directory="ycv/"
):
    """
    Display an image in a resizable window while maintaining aspect ratio.

    Parameters:
        image (numpy.ndarray): Input image array in BGR format.
        title (str, optional): Window title for display. Defaults to 'Image'
        max_height (int, optional): Maximum height of display window in pixels. Defaults to 720.
        show_in_separate_window (bool, optional): If True, displays the image in a separate window. If False, displays the image inline using Matplotlib plot (for jupyter). Defaults to True.
        dpi (int, optional): Dots per inch for Matplotlib display, affects figure size (default is 100).
    Returns:
        None
    Notes:
    - OpenCV window (cv2.namedWindow) Does not run properly on WSL.
    - For OpenCV window (show_in_separate_window=True):
        * Press any key to close the window
        * Window is resizable while maintaining aspect ratio
    - For Matplotlib display (show_in_separate_window=False):
        * Image is converted from BGR to RGB color space
        * Figure size is calculated based on max_height and aspect ratio
    Examples:
    >>> img = cv2.imread('image.jpg')
    >>> show(img, title='My Image', max_height=600)  # OpenCV window
    >>> show(img, show_in_separate_window=False)     # Matplotlib display
    """
    # Get the dimensions of the image
    if len(image.shape) == 2:
        height, width = image.shape
    elif len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        raise ValueError("Invalid image shape. Expected 2D or 3D array.")
    # Calculate the aspect ratio of the original image
    aspect_ratio = width / height
    # Calculate the corresponding width to maintain the aspect ratio
    max_width = int(max_height * aspect_ratio)
    if show_in_separate_window:

        # Create a window with a specific size
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            title, max_width, max_height
        )  # Set the window size to  max_width * max_height

        # show the original image in the smaller window
        cv2.imshow(title, image)
        # Wait for a key press
        cv2.waitKey(0)
        if save:
            # Save the image to a file
            cv2.imwrite(f"{directory}{title}.png", image)
        # Close all windows
        cv2.destroyAllWindows()
        return None

    # Set figure size in inches and DPI
    set_figsize_pixels = lambda width, height, dpi: (width / dpi, height / dpi)
    # Note that the figsize parameter is in inches, so we need to divide the pixel values by the DPI to get the correct size in inches.
    # actual figure size may vary slightly depending on the display and rendering settings.
    fig = plt.figure(figsize=set_figsize_pixels(max_width, max_height, dpi), dpi=dpi)

    # Set the title
    plt.title(title)
    plt.axis("off")
    if len (image.shape) == 2:
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if save:
        # Save the figure to a file
        plt.savefig(f"{directory}{title}.png", dpi=dpi, bbox_inches="tight")
    if show_in_line:
        plt.show()
    plt.clf()

    return None


def get_mass_type(image_name, return_binary: False):
    """
    This function takes an image name as input and returns the mass type (label) based on the image name.

    Parameters:
        - image_name (str): The name of the breast mass image.

    Returns:
        str: Ground truth mass type from the annotations file.

    Raises:
        ValueError: If no record is found for the given image_name.
    """
    label_map = {
        "Normal": -1, # For images having segmentation mask(s) with "normal" label.
        "Benign": 0 ,
        "Malignant": 1,
    }

    # Filter the DataFrame by the given image_name
    dataset_df = pd.read_csv(ProjectPaths.annotations_all_sheet_modified)
    # dataset_df = pd.read_excel(ProjectPaths.annotations, sheet_name="all")
    # dataset_df["Image_name"] could have trailing whitespace.
    # dataset_df["Image_name"] = dataset_df["Image_name"].str.strip()
    filtered_row = dataset_df[dataset_df["Image_name"] == image_name]

    if not filtered_row.empty:
        # Extract the label from the filtered row (assuming the label is stored in a column named 'label')
        label = filtered_row.iloc[0]["Pathology Classification/ Follow up"]
        if return_binary:
            return label_map[label]
        return label
    else:
        # If no record is found for the given image_name, raise value error.
        raise ValueError(f"No record found for image name {image_name}")


def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a grayscale image.
    For 3-channel grayscale images, takes one channel, applies CLAHE, and repeats the result.
    
    Args:
        image: Input image (single or 3-channel grayscale)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        3-channel image with CLAHE applied
    """
    # Create CLAHE object
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Handle 3-channel grayscale images
    if len(image.shape) == 3:
        # Take just one channel since they're all the same
        single_channel = image[:, :, 0]
        # Apply CLAHE to single channel
        enhanced_channel = clahe_obj.apply(single_channel)
        # return enhanced_channel
        # Stack the enhanced channel three times
        return cv2.merge([enhanced_channel, enhanced_channel, enhanced_channel])
    else:
        # For single channel images
        return clahe_obj.apply(image)


def min_max_normalise(image, low=0.0, high=1.0):
    """
    Apply min-max normalization to an image. maps the pixel values to a specified range [low, high].
    For 3-channel grayscale images, takes one channel, normalizes it, and repeats the result.
    input image is expected to be in float32 format.
    """
    if len(image.shape) == 3:
        single_channel = image[:, :, 0]
        normalized_channel = cv2.normalize(
            single_channel.astype('float32'), None, alpha=low, beta=high, norm_type=cv2.NORM_MINMAX
            )
        return cv2.merge([normalized_channel, normalized_channel, normalized_channel])
    else:
        normalized_image = cv2.normalize(
            image.astype('float32'), None, alpha=low, beta=high, norm_type=cv2.NORM_MINMAX
            )
        return normalized_image


class Chronometer:
    """Chronometer obj represents a chronometer irl."""

    def __init__(self, log_name: str, lap_name="lap"):
        """constructor.

        Args:
            log_name (str): filename for log file. will be saved as 'filename_chronometer.log'
            lap_name (str, optional): name of the lap for logging. Defaults to "lap".

        Raises:
            ValueError: if log_name contains dot(.).
        """
        if "." in log_name:
            raise ValueError(f"expected str that has no dots(.). got: {log_name}")

        self.__logger = logging.getLogger("Chronometer")
        self.__logger.setLevel(logging.INFO)
        # Create console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        # Create formatter and add it to handler
        c_format = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s - %(message)s", datefmt="%H:%M:%S"
        )
        c_handler.setFormatter(c_format)
        # add handler to logger
        self.__logger.addHandler(c_handler)

        # create logs directory and file handler
        try:
            os.makedirs(os.path.dirname("./logs/"), exist_ok=True)
        except Exception:
            # does logging raise exceptions?
            f_handler = logging.FileHandler(f"./{log_name}_chronometer.log")
            self.__logger.exception(
                "unable to make 'logs' directory. logs will be save in root"
            )
        else:
            f_handler = logging.FileHandler(f"./logs/{log_name}_chronometer.log")
            self.__logger.warning("logs will be saved in 'logs' directory.")

        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter(
            "%(asctime)s, %(levelname)s, %(message)s", datefmt="%y-%b-%d %H:%M:%S"
        )
        f_handler.setFormatter(f_format)
        self.__logger.addHandler(f_handler)

        self.__tiks = [time.time()]
        self.lap_name = lap_name
        self.__lap_count = 0

    @property
    def lap_name(self):
        """get lap_name.

        Returns:
            str: lap_name. Defaults to "lap".
        """
        return self._lap_name

    @lap_name.setter
    def lap_name(self, name: str):
        """set lap name.

        Args:
            name (str): lap name

        Raises:
            ValueError: if name arg is empty.
        """
        if not name:
            raise ValueError("missing name")
        self.__lap_count = 0
        self._lap_name = name

    def __del__(self):
        """deconstructor"""
        self.__logger.warning("was deconstructed.")

    def lap(self) -> None:
        """log lap duration."""
        self.__tiks.append(time.time())
        self.__lap_count += 1
        self.__logger.info(
            "%s %s: %s",
            self.lap_name,
            self.__lap_count,
            Chronometer.__hms_string(self.__tiks[-1] - self.__tiks[-2]),
        )

    def get_durations(self) -> list:
        """get recorded durations in seconds, as list of floats.

        Returns:
            list: a list of single decimal, float numbers.
        """
        return [
            round(j - i, ndigits=1) for i, j in zip(self.__tiks[:-1], self.__tiks[1:])
        ]

    def avg_lap_time(self) -> None:
        """log average lap time."""
        durations = self.get_durations()
        if durations:
            self.__logger.info(
                "average duration: %s",
                Chronometer.__hms_string(sum(durations) / len(durations)),
            )
        else:
            self.__logger.warning("chronometer has only recorded one time stamp")

    def get_total_time(self) -> str:
        """get total time spent. difference between obj's construction time stamp and latest lap.

        Returns:
            str: hh:mm:ss
        """
        return f"elapsed time: {Chronometer.__hms_string(self.__tiks[-1] - self.__tiks[0])}"

    def reset(self) -> None:
        """reset chronometer obj as of new. Can beep as a sign.

        Args:
            n_beeps (int, optional): number of times to beep. Defaults to 0.
        """
        self.__logger.warning("was reset. %s", self.get_total_time())
        self.__tiks.clear()
        self.__tiks = [time.time()]
        self.__lap_count = 0

    @staticmethod
    def __hms_string(sec_elapsed: float) -> str:
        """Nicely formatted time string.

        Args:
            sec_elapsed (float): expects time.time().

        Returns:
            str: hh:mm:ss
        """
        hour = int(sec_elapsed / (60 * 60))
        minute = int((sec_elapsed % (60 * 60)) / 60)
        sec = sec_elapsed % 60
        return f"{hour}:{minute:>02}:{sec:>02.0f}"

def update_images_absolute_paths():
    """
    updates images absolute paths in Radiology_manual_annotations_all_sheet_modified file.

    Parameters:
        None

    Returns:
        None
    """
    
    # TODO: make sure to strip trailing spaces from image names.
    dataset_df = pd.read_excel(ProjectPaths.det_annotations_org, sheet_name="all")

    dataset_df["Absolute_path"] = dataset_df["Image_name"].apply(
        lambda x: (
            os.path.join(ProjectPaths.subtracted_images, x.strip() + ".jpg")
            if "_CM_" in x
            else os.path.join(ProjectPaths.low_energy_images, x.strip() + ".jpg")
        )
    )
    # TODO: make sure to strip trailing spaces from image names in "Image_name" column.
    dataset_df["Image_name"] = dataset_df["Image_name"].str.strip()
    # Specify the file path where you want to save the CSV file and save this dataframe
    dataset_df.to_csv(ProjectPaths.annotations_all_sheet_modified, index=False)
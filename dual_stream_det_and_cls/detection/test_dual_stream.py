import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .dual_stream_retinanet import DualStreamRetinaNet
from ..utils import cvtColor, resize_image
from .utils.utils_bbox import decodebox, non_max_suppression
from .utils.utils_map import get_map
from ..immutables import Hyperparameter, ProjectPaths
from ..medical_image_utils import min_max_normalise
class DualStreamDetector(object):
    """
    Class to perform inference using a trained DualStreamRetinaNet model.
    """
    _defaults = {
        "model_path"        : ProjectPaths.best_and_last_det_model + '/fold_4/s2_best_ep029-loss0.003-val_loss0.408-val_map0.630.pth',
        "input_shape"       : Hyperparameter.input_shape,
        "confidence"        : 0.0, 
        "draw_confidence"   : 0.5, # Confidence threshold for drawing bounding boxes.
        "nms_iou"           : 0.5,
        "letterbox_image"   : False, # Simple resize was used in training
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        # num_classes will be 1 for objectness (mass)
        self.num_classes = Hyperparameter.num_classes

        self.generate()

    def generate(self):
        """
        Loads the model and weights.
        """
        print('Loading weights into state dict...')
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        
        self.model = DualStreamRetinaNet(num_classes=self.num_classes, phi=2)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        self.model = self.model.eval()
        print(f'{self.model_path} model loaded.')

        if self.cuda:
            self.model = self.model.to(self.device)

    def _draw_predictions(self, image: Image.Image, top_boxes: list, top_conf: list, gt_boxes: list = None) -> Image.Image:
        """
        Draws bounding boxes and classification labels on an image.

        Args:
            image (Image.Image): The PIL Image to draw on.
            top_boxes (list): List of predicted bounding box coordinates.
            top_conf (list): List of confidence scores for each predicted box.
            image_label (str): The overall predicted label for the image ('Benign' or 'Malignant').
            display_prob (float): The probability associated with the image label.
            label_color (tuple): The RGB color for the label text.
            image_prefix (str): A prefix to add to the classification text (e.g., "DM - ").
            gt_boxes (list, optional): List of ground truth bounding boxes. Defaults to None.
            gt_label_str (str, optional): The ground truth label string. Defaults to None.

        Returns:
            Image.Image: The image with predictions and ground truth drawn on it.
        """
        draw = ImageDraw.Draw(image)
        image_height, image_width = np.array(np.shape(image)[0:2])
        
        # Setup for drawing
        min_dimension = min(image.size[0], image.size[1])
        font_size = int(min_dimension * 0.025)  # 2.5% of smallest image dimension
        try:
            font = ImageFont.truetype(font=ProjectPaths.font, size=font_size)
        except IOError:
            print(f"Font file not found. Using default font.")
            font = ImageFont.load_default()
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # Draw ground truth bounding boxes in magenta
        if gt_boxes:
            gt_box_color = (255, 0, 255) # Magenta
            for box in gt_boxes:
                left, top, right, bottom = box # x_min, y_min, x_max, y_max
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image_height, np.floor(bottom).astype('int32'))
                right   = min(image_width, np.floor(right).astype('int32'))
                for j in range(thickness):
                    draw.rectangle([left + j, top + j, right - j, bottom - j], outline=gt_box_color)

        # Draw bounding boxes for detected masses
        for i, box in enumerate(top_boxes):
            score = top_conf[i]
            if score < self.draw_confidence:
                continue
            top, left, bottom, right = box

            # Ensure coordinates are within image bounds
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image_height, np.floor(bottom).astype('int32'))
            right   = min(image_width, np.floor(right).astype('int32'))

            # Draw the box
            box_color = (255, 255, 0) # Yellow
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=box_color)
            
            # Prepare and draw the confidence score label
            label = f'{score:.2f}'
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            
            # Position text above the box if there's space, otherwise inside
            is_score_above = top - text_height - 10 >= 0
            text_y = top - text_height - 10 if is_score_above else top
            text_x = left - 2 if is_score_above else left + 4
            draw.text((text_x, text_y), label, fill=box_color, font=font)
            
        del draw
        return image

    def detect_image(self, dm_image_path: str, cm_image_path: str, gt_boxes: list = None) -> tuple:
        """
        Detects objects and classifies the image pair, returning a combined image and raw predictions.

        Args:
            dm_image_path (str): Path to the low-energy (DM) image.
            cm_image_path (str): Path to the contrast-enhanced (CM) image.
            gt_boxes (list, optional): List of ground truth bounding boxes. Defaults to None.
            gt_label_str (str, optional): The ground truth label string. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - PIL.Image: A single image with the annotated DM and CM images placed side-by-side, or None on error.
                - str: The predicted image label ('Benign' or 'Malignant'), or None on error.
                - list: A list of predicted bounding box coordinates.
                - list: A list of confidence scores for the predicted boxes.
        """
        # --- 1. Load Images ---
        try:
            dm_image = Image.open(dm_image_path)
            cm_image = Image.open(cm_image_path)
        except FileNotFoundError:
            print(f'Error: Cannot open image file. Check paths: {dm_image_path}, {cm_image_path}')
            return None, None, [], []

        # Store original images for drawing
        dm_image_to_draw = dm_image.copy()
        cm_image_to_draw = cm_image.copy()
        image_height, image_width = np.array(np.shape(dm_image_to_draw)[0:2])

        # --- 2. Preprocess Images ---
        dm_image_data = self._preprocess_image(dm_image)
        cm_image_data = self._preprocess_image(cm_image)
        
        # --- 3. Run Inference ---
        with torch.no_grad():
            images_dm = torch.from_numpy(dm_image_data).type(torch.FloatTensor).to(self.device)
            images_cm = torch.from_numpy(cm_image_data).type(torch.FloatTensor).to(self.device)

            start_time = time.time()
            regression, classification, anchors = self.model(images_dm, images_cm, cuda=self.cuda)
            inference_time = time.time() - start_time
            # print(f"Inference time: {inference_time:.4f} seconds.")
            
            # --- 4. Post-process Results ---
            decoded_boxes = decodebox(regression[0], anchors[0], self.input_shape)
            prediction_tensor = torch.cat([decoded_boxes, classification[0]], axis=-1)

            results = non_max_suppression(
                prediction_tensor, self.input_shape, (image_height, image_width),
                self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou
            )

        # --- 5. Visualize Results ---
        if results[0] is not None:
            top_conf = results[0][:, 4].tolist()
            top_boxes = results[0][:, :4].tolist()
        else:
            top_conf, top_boxes = [], []
            
        # Draw on both DM and CM images
        dm_image_to_draw = self._draw_predictions(dm_image_to_draw, top_boxes, top_conf,gt_boxes=gt_boxes)
        cm_image_to_draw = self._draw_predictions(cm_image_to_draw, top_boxes, top_conf, gt_boxes=gt_boxes)

        # --- 6. Combine Images ---
        total_width = dm_image_to_draw.width + cm_image_to_draw.width
        max_height = max(dm_image_to_draw.height, cm_image_to_draw.height)

        combined_image = Image.new('RGB', (total_width, max_height))
        combined_image.paste(dm_image_to_draw, (0, 0))
        combined_image.paste(cm_image_to_draw, (dm_image_to_draw.width, 0))
        
        # return combined_image, image_label, top_boxes, top_conf
        return combined_image, top_boxes, top_conf

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesses a single image for model input.
        """
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # Add batch dimension and channel-first format
        image_data = np.expand_dims(np.transpose(min_max_normalise(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        return image_data

if __name__ == "__main__":
    detector = DualStreamDetector()
    
    test_on_train = False
    test_annotation_path = ProjectPaths.val_annotations_fold_4
    try:
        os.mkdir(ProjectPaths.predictions)
        print(f"Directory '{ProjectPaths.predictions}' created successfully.")
    except FileExistsError:
        print(f"Directory '{ProjectPaths.predictions}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{ProjectPaths.predictions}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # --- Setup for Evaluation ---
    map_out_path = ProjectPaths.dual_stream_det + "/map_out"
    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)

    try:
        os.makedirs(map_out_path + "/detection-results")
    except PermissionError:
        print(f"Permission denied: Unable to create '{map_out_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
  
    os.makedirs(map_out_path + "/ground-truth")
    
    detection_class_name = "mass"

    print(f"Reading test cases from: {test_annotation_path}")
    
    try:
        with open(test_annotation_path, 'r') as f:
            test_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Test annotation file not found at {test_annotation_path}")
        exit()
        
    for line in tqdm(test_lines, desc="Processing Test Images for Evaluation"):
        parts = line.strip().split()
        if not parts:
            continue
            
        # 1. Parse annotations
        image_name_from_file = parts[0]
        if test_on_train and ("_resized" not in image_name_from_file):
            continue
        
        gt_boxes = []
        if len(parts) > 2:
            for bbox_str in parts[2:]:
                coords = list(map(int, bbox_str.split(',')))
                gt_boxes.append(coords[:4])
        
        image_id = Path(image_name_from_file).stem

        # 2. Construct image paths
        if "_DM_" in image_name_from_file:
            dm_filename = image_name_from_file
            cm_filename = image_name_from_file.replace("_DM_", "_CM_")
        else: # Assume "_CM_"
            cm_filename = image_name_from_file
            dm_filename = image_name_from_file.replace("_CM_", "_DM_")
            
        dm_path = ProjectPaths.det_dataset + "/" + dm_filename
        cm_path = ProjectPaths.det_dataset + "/" + cm_filename
        
        # 3. Perform detection and classification
        result_image, pred_boxes, pred_confs = detector.detect_image(
            str(dm_path), 
            str(cm_path),
            gt_boxes=gt_boxes,
        )
        
        if result_image is None:
            continue
        
        # 4. Write files for mAP calculation
        with open(map_out_path + "/ground-truth/" + f"{image_id}.txt", "w") as f:
            for box in gt_boxes:
                left, top, right, bottom = box
                f.write(f"{detection_class_name} {left} {top} {right} {bottom}\n")
        
        with open(map_out_path + "/detection-results/" + f"{image_id}.txt", "w") as f:
            for i, box in enumerate(pred_boxes):
                top, left, bottom, right = box
                score = pred_confs[i]
                f.write(f"{detection_class_name} {score} {int(left)} {int(top)} {int(right)} {int(bottom)}\n")

        # 5. Save the visual result
        base_name = image_id.replace("_DM", "").replace("_CM", "")
        output_path = ProjectPaths.predictions + f"/pred_{base_name}.jpg"
        result_image.save(output_path)

    # --- 6. Calculate and Report Metrics ---
    print("\n--- Evaluation Results ---")

    # Detection Metrics (AP)
    print("Calculating Average Precision (AP) for 'mass' detection...")
    get_map(MINOVERLAP=0.5, draw_plot=False, path=str(map_out_path))

    mass_ap = 0.0
    try:
        with open(map_out_path + "/results/results.txt", "r") as f:
            for line in f:
                if "mass AP" in line:
                    mass_ap = float(line.split('%')[0]) / 100.0
                    break
        det_results_str = (
            f"Object Detection Metrics:\n"
            f"----------------------------------------------------------\n"
            f"  - Average Precision (AP) for 'mass' class (IoU=0.5): {mass_ap:.4f}\n"
            f"----------------------------------------------------------\n"
        )
    except Exception as e:
        det_results_str = (
            f"Object Detection Metrics:\n"
            f"----------------------------------------------------------\n"
            f"  - Could not calculate AP. Error: {e}\n"
            f"----------------------------------------------------------\n"
        )
    print(det_results_str)
    
    # Save results to a file
    results_file_path = ProjectPaths.predictions + "/evaluation_results.txt"
    with open(results_file_path, "w") as f:
        f.write("--- Evaluation Results ---\n\n")
        # f.write(cls_results_str)
        f.write(det_results_str)
    print(f"Evaluation results saved to {results_file_path}")

    # Clean up
    shutil.rmtree(map_out_path)
    print(f"Cleaned up temporary directory: {map_out_path}")
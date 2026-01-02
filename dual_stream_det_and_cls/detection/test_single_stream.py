import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .single_stream_retinanet import SingleStreamRetinaNet
from ..utils import cvtColor, resize_image
from .utils.utils_bbox import decodebox, non_max_suppression
from .utils.utils_map import get_map
from ..immutables import Hyperparameter, ProjectPaths
from ..medical_image_utils import min_max_normalise

# Configuration
# Options: 'dm', 'cm', 'both', 'dual_channel'
STREAM_MODE = 'dual_channel' 
VAL_ANN_PATH = ProjectPaths.det_dataset + "/val_annotations_fold_4.txt"
# Update model path to point to your trained single stream model
MODEL_PATH = ProjectPaths.det_training_logs + '/single_stream/dual_channel_synergistic_baseAugs_elastic_gridshuffle_cutout_no_clahe/single_stream_fold_4/s2_best_ep015-loss0.030-val_loss0.192-val_map0.386.pth'

class SingleStreamDetector(object):
    """
    Class to perform inference using a trained SingleStreamRetinaNet model.
    Supports 'dm', 'cm', 'both', and 'dual_channel' inference modes.
    """
    _defaults = {
        "model_path"        : MODEL_PATH,
        "input_shape"       : Hyperparameter.input_shape,
        "confidence"        : 0.0, 
        "draw_confidence"   : 0.5,
        "nms_iou"           : 0.5,
        "letterbox_image"   : False,
        "cuda"              : True,
        "mode"              : STREAM_MODE
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
        
        self.num_classes = Hyperparameter.num_classes
        self.generate()

    def generate(self):
        """Loads the model and weights."""
        print(f'Loading weights into state dict for mode: {self.mode}...')
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.cuda else 'cpu')
        
        self.model = SingleStreamRetinaNet(num_classes=self.num_classes, phi=2)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        self.model = self.model.eval()
        print(f'{self.model_path} model loaded.')

        if self.cuda:
            self.model = self.model.to(self.device)

    def _draw_predictions(self, image: Image.Image, top_boxes: list, top_conf: list, gt_boxes: list = None) -> Image.Image:
        """Draws bounding boxes and labels on an image."""
        draw = ImageDraw.Draw(image)
        image_height, image_width = np.array(np.shape(image)[0:2])
        
        min_dimension = min(image.size[0], image.size[1])
        font_size = int(min_dimension * 0.025)
        try:
            font = ImageFont.truetype(font=ProjectPaths.font, size=font_size)
        except IOError:
            font = ImageFont.load_default()
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # Draw ground truth
        if gt_boxes:
            gt_box_color = (255, 0, 255) # Magenta
            for box in gt_boxes:
                left, top, right, bottom = box
                top, left = max(0, int(top)), max(0, int(left))
                bottom, right = min(image_height, int(bottom)), min(image_width, int(right))
                for j in range(thickness):
                    draw.rectangle([left + j, top + j, right - j, bottom - j], outline=gt_box_color)

        # Draw predictions
        for i, box in enumerate(top_boxes):
            score = top_conf[i]
            if score < self.draw_confidence:
                continue
            top, left, bottom, right = box
            top, left = max(0, int(top)), max(0, int(left))
            bottom, right = min(image_height, int(bottom)), min(image_width, int(right))

            box_color = (255, 255, 0) # Yellow
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=box_color)
            
            label = f'{score:.2f}'
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            
            is_score_above = top - text_height - 10 >= 0
            text_y = top - text_height - 10 if is_score_above else top
            text_x = left - 2 if is_score_above else left + 4
            draw.text((text_x, text_y), label, fill=box_color, font=font)
            
        del draw
        return image

    def _prepare_dual_channel_input(self, dm_path, cm_path):
        """Prepares input for the specific 'dual_channel' single stream mode."""
        try:
            dm_image = Image.open(dm_path).convert('L')
            cm_image = Image.open(cm_path).convert('L')
        except FileNotFoundError:
            return None, None

        h, w = self.input_shape
        dm_resized = dm_image.resize((w, h), Image.BICUBIC)
        cm_resized = cm_image.resize((w, h), Image.BICUBIC)

        combined_data = np.zeros((h, w, 3), dtype=np.float32)
        combined_data[..., 0] = np.array(dm_resized, dtype=np.float32)
        combined_data[..., 1] = np.array(cm_resized, dtype=np.float32)
        
        # Normalize and transpose to (C, H, W)
        image_data = np.transpose(min_max_normalise(combined_data), (2, 0, 1))
        image_data = np.expand_dims(image_data, 0) # Add batch dim
        return image_data, cm_image # Return one PIL image for visualization reference

    def _prepare_standard_input(self, image_path):
        """Prepares input for 'dm', 'cm', or 'both' modes (RGB/Grayscale)."""
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            return None, None
        
        image_rgb = cvtColor(image)
        image_data = resize_image(image_rgb, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.transpose(min_max_normalise(np.array(image_data, dtype='float32')), (2, 0, 1))
        image_data = np.expand_dims(image_data, 0)
        return image_data, image

    def detect_image(self, image_name: str, gt_boxes: list = None) -> tuple:
        """
        Detects objects based on the configured STREAM_MODE.
        Args:
            image_name: The base filename (e.g., P1_L_DM_MLO.jpg). 
                        For dual_channel, assumes finding the partner.
        """
        
        input_tensor = None
        visual_image = None
        
        # 1. Prepare Input based on Mode
        if self.mode == 'dual_channel':
            # Construct paths for both DM and CM
            if "_CM_" in image_name:
                cm_path = os.path.join(ProjectPaths.det_dataset, image_name)
                dm_path = os.path.join(ProjectPaths.det_dataset, image_name.replace("_CM_", "_DM_"))
            else:
                dm_path = os.path.join(ProjectPaths.det_dataset, image_name)
                cm_path = os.path.join(ProjectPaths.det_dataset, image_name.replace("_DM_", "_CM_"))
            
            input_tensor, visual_image = self._prepare_dual_channel_input(dm_path, cm_path)
            # For visualization in dual channel, we usually just show one or combine them visually
            # converting visual_image back to RGB for drawing
            if visual_image: visual_image = visual_image.convert("RGB") 

        else:
            # Standard single image mode (dm, cm, or both)
            image_path = os.path.join(ProjectPaths.det_dataset, image_name)
            input_tensor, visual_image = self._prepare_standard_input(image_path)

        if input_tensor is None:
            print(f"Error: Could not process image {image_name}")
            return None, [], []

        image_height, image_width = np.array(np.shape(visual_image)[0:2])

        # 2. Run Inference
        with torch.no_grad():
            images = torch.from_numpy(input_tensor).type(torch.FloatTensor).to(self.device)
            
            start_time = time.time()
            regression, classification, anchors = self.model(images, cuda=self.cuda)
            # inference_time = time.time() - start_time
            
            # 3. Post-process
            decoded_boxes = decodebox(regression[0], anchors[0], self.input_shape)
            prediction_tensor = torch.cat([decoded_boxes, classification[0]], axis=-1)

            results = non_max_suppression(
                prediction_tensor, self.input_shape, (image_height, image_width),
                self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou
            )

        # 4. Extract Results
        if results[0] is not None:
            top_conf = results[0][:, 4].tolist()
            top_boxes = results[0][:, :4].tolist()
        else:
            top_conf, top_boxes = [], []
            
        # 5. Draw
        if visual_image:
            visual_image = self._draw_predictions(visual_image, top_boxes, top_conf, gt_boxes=gt_boxes)

        return visual_image, top_boxes, top_conf

if __name__ == "__main__":
    detector = SingleStreamDetector()
    
    try:
        os.makedirs(ProjectPaths.predictions, exist_ok=True)
        print(f"Directory '{ProjectPaths.predictions}' ready.")
    except Exception as e:
        print(f"An error occurred creating output directory: {e}")

    # --- Setup for Evaluation ---
    map_out_path = ProjectPaths.dual_stream_det + "/map_out_single_stream"
    if os.path.exists(map_out_path):
        shutil.rmtree(map_out_path)
    os.makedirs(map_out_path + "/detection-results", exist_ok=True)
    os.makedirs(map_out_path + "/ground-truth", exist_ok=True)
    
    detection_class_name = "mass"

    print(f"Reading test cases from: {VAL_ANN_PATH}")
    try:
        with open(VAL_ANN_PATH, 'r') as f:
            test_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Test annotation file not found at {VAL_ANN_PATH}")
        exit()

    # --- Filter Lines Based on Mode ---
    # Create a unique list of items to process based on the selected mode
    items_to_process = {} 
    
    for line in test_lines:
        parts = line.strip().split()
        if not parts: continue
        
        image_name = parts[0]
        gt_data = [list(map(int, box.split(',')))[:4] for box in parts[2:]]
        
        should_process = False
        key = image_name 

        if STREAM_MODE == 'dual_channel':
            # Normalize key to ensure we process the pair only once
            key = image_name.replace("_CM_", "_").replace("_DM_", "_")
            should_process = True # We process unique keys
        elif STREAM_MODE == 'dm':
            if "_DM_" in image_name:
                should_process = True
            elif "_CM_" in image_name:
                # Need to check if we should infer the DM counterpart
                # Usually annotations list distinct files. If file is CM, skip for DM mode
                # unless we want to force evaluation on the DM partner of a CM entry.
                # Assuming annotation file lists specific images to test:
                pass 
        elif STREAM_MODE == 'cm':
            if "_CM_" in image_name:
                should_process = True
        elif STREAM_MODE == 'both':
            should_process = True
        
        # If strict matching based on filename presence in annotation file:
        if should_process:
            if key not in items_to_process:
                items_to_process[key] = {
                    'image_name': image_name,
                    'gt_boxes': gt_data
                }
            # For dual channel, if we hit the second file of the pair, we don't overwrite 
            # because the GT is the same and we only need one trigger.

    print(f"Mode '{STREAM_MODE}': Found {len(items_to_process)} unique items to evaluate.")

    # --- Inference Loop ---
    for key, data in tqdm(items_to_process.items(), desc=f"Processing {STREAM_MODE}"):
        image_name_from_file = data['image_name']
        gt_boxes = data['gt_boxes']
        
        # For dual_channel, image_name_from_file is just one of the pair, 
        # detect_image handles finding the partner.
        result_image, pred_boxes, pred_confs = detector.detect_image(
            image_name_from_file,
            gt_boxes=gt_boxes,
        )
        
        if result_image is None:
            continue
        
        image_id = Path(image_name_from_file).stem
        # Adjust ID for dual channel to represent the pair
        if STREAM_MODE == 'dual_channel':
            image_id = image_id.replace("_CM_", "_").replace("_DM_", "_")

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
        output_path = ProjectPaths.predictions + f"/ss_{STREAM_MODE}_{image_id}.jpg"
        result_image.save(output_path)

    # --- 6. Calculate and Report Metrics ---
    print("\n--- Evaluation Results ---")

    def extract_metrics(results_path, class_name="mass"):
        metrics = {}
        try:
            with open(results_path, "r") as f:
                for line in f:
                    if f"{class_name} AP" in line:
                        metrics['AP'] = float(line.split('%')[0]) / 100.0
                    elif f"{class_name} F1-score" in line:
                        metrics['F1'] = float(line.split(':')[1].strip()) / 100.0
                    elif f"{class_name} Recall" in line:
                        metrics['Recall'] = float(line.split(':')[1].strip()) / 100.0
                    elif f"{class_name} Precision" in line:
                        metrics['Precision'] = float(line.split(':')[1].strip()) / 100.0
        except Exception:
            pass
        return metrics

    ap_results = {}
    detailed_metrics = {}

    results_dir = os.path.join(map_out_path, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # AP at MINOVERLAP=0
    print("Calculating Average Precision (AP) for 'mass' detection (MINOVERLAP=0)...")
    get_map(MINOVERLAP=0, draw_plot=False, path=str(map_out_path))
    metrics_0 = extract_metrics(os.path.join(results_dir, "results.txt"))
    ap_results["AP0"] = metrics_0.get('AP')
    detailed_metrics["0.00"] = metrics_0

    os.makedirs(results_dir, exist_ok=True)
    
    # AP50 (MINOVERLAP=0.50)
    print("Calculating AP50 (MINOVERLAP=0.50)...")
    get_map(MINOVERLAP=0.50, draw_plot=False, path=str(map_out_path))
    metrics_50 = extract_metrics(os.path.join(results_dir, "results.txt"))
    ap_results["AP50"] = metrics_50.get('AP')
    detailed_metrics["0.50"] = metrics_50

    # AP@[.50:.05:.95]
    ap_list = []
    for ov in np.arange(0.50, 0.96, 0.05):
        print(f"Calculating AP at MINOVERLAP={ov:.2f}...")
        os.makedirs(results_dir, exist_ok=True)
        get_map(MINOVERLAP=ov, draw_plot=False, path=str(map_out_path))
        metrics = extract_metrics(os.path.join(results_dir, "results.txt"))
        if metrics.get('AP') is not None:
            ap_list.append(metrics.get('AP'))
        detailed_metrics[f"{ov:.2f}"] = metrics
    
    if ap_list:
        ap_results["AP@[.50:.05:.95]"] = np.mean(ap_list)
    else:
        ap_results["AP@[.50:.05:.95]"] = None

    # --- Reporting ---
    det_results_str = (
        f"Single Stream ({STREAM_MODE}) Object Detection Metrics:\n"
        f"----------------------------------------------------------\n"
        f"  - AP (MINOVERLAP=0): {ap_results['AP0']:.4f}\n"
        f"  - AP50 (MINOVERLAP=0.50): {ap_results['AP50']:.4f}\n"
        f"  - AP@[.50:.05:.95]: {ap_results['AP@[.50:.05:.95]']:.4f}\n"
        f"----------------------------------------------------------\n\n"
        f"Detailed Metrics for Each IoU Threshold:\n"
        f"----------------------------------------------------------\n"
    )

    for threshold, metrics in sorted(detailed_metrics.items()):
        det_results_str += (
            "IoU Threshold = {}:\n"
            "  - AP: {:.4f}\n"
            "  - F1-score: {:.4f}\n"
            "  - Recall: {:.4f}\n"
            "  - Precision: {:.4f}\n"
            "----------------------------------------------------------\n"
        ).format(
            threshold,
            metrics.get('AP', 0.0),
            metrics.get('F1', 0.0),
            metrics.get('Recall', 0.0),
            metrics.get('Precision', 0.0)
        )

    print(det_results_str)

    results_file_path = ProjectPaths.predictions + f"/evaluation_results_ss_{STREAM_MODE}.txt"
    with open(results_file_path, "w") as f:
        f.write(f"--- Evaluation Results ({STREAM_MODE}) ---\n\n")
        f.write(det_results_str)
    print(f"Evaluation results saved to {results_file_path}")

    shutil.rmtree(map_out_path)
    print(f"Cleaned up temporary directory: {map_out_path}")
"""
Unified script that combines dataset preparation and YOLOv8 training for cross-validation.
The script handles:
1. Converting annotations to YOLO format
2. Creating dataset structure
3. Training YOLOv8 model with cross-validation
"""
import os
import shutil
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
import torch
from tqdm import tqdm

from ...immutables import ProjectPaths

# --- Configuration ---
NUM_FOLDS = 5
STREAM_MODE = 'dm'  # Options: 'cm', 'dm', 'both'
EPOCHS = 30
BATCH_SIZE = -1
IMG_SIZE = 1280
MODEL_VARIANT = 'yolov8l.pt'
PROJECT_NAME = ProjectPaths.yolo_dataset + f'/results/{MODEL_VARIANT.strip(".pt")}_5-Fold_CV'
FREEZE_LAYERS = 0 # Freeze first x layers
IMG_DIMS = (1280, 1280)  # Dimensions of input images

def convert_to_yolo_format(box, img_width, img_height):
    """Convert Pascal VOC box format to YOLO format."""
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return x_center, y_center, width, height

def prepare_fold_data(input_ann_file, output_dir, set_name, stream_mode='dm', img_dims=IMG_DIMS):
    """Process annotation file and organize data into YOLO format."""
    print(f"Preparing '{set_name}' set for stream '{stream_mode}' from {Path(input_ann_file).name}...")
    
    img_dir = os.path.join(output_dir, set_name, 'images')
    label_dir = os.path.join(output_dir, set_name, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    with open(input_ann_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Processing {set_name} data"):
        parts = line.strip().split()
        if not parts:
            continue
        
        image_name = parts[0]
        
        # Stream filtering
        is_cm = "_CM_" in image_name
        is_dm = "_DM_" in image_name
        
        if stream_mode == 'cm' and not is_cm:
            continue
        if stream_mode == 'dm' and not is_dm:
            continue
        
        source_img_path = os.path.join(ProjectPaths.det_dataset, image_name)
        if not os.path.exists(source_img_path):
            print(f"Warning: Image not found, skipping: {source_img_path}")
            continue
        
        target_img_path = os.path.join(img_dir, image_name)
        shutil.copy(source_img_path, target_img_path)
        
        # Convert annotations
        img_width, img_height = img_dims
        boxes = [list(map(int, box.split(','))) for box in parts[2:]]
        
        label_file_path = os.path.join(label_dir, Path(image_name).stem + '.txt')
        with open(label_file_path, 'w') as lf:
            for box in boxes:
                class_id = box[4]
                pascal_box = box[:4]
                yolo_box = convert_to_yolo_format(pascal_box, img_width, img_height)
                lf.write(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")

def run_fold_training(fold_k: int, stream_mode: str):
    """Prepare data and run YOLOv8 training for a single fold."""
    print(f"\n{'='*30}\nRUNNING FOLD {fold_k} | STREAM MODE: {stream_mode}\n{'='*30}")

    # 1. Prepare Dataset
    yolo_data_dir = os.path.join(ProjectPaths.yolo_dataset, f'yolo_dataset_fold_{fold_k}_{stream_mode}')
    
    # Clean up previous data
    if os.path.exists(yolo_data_dir):
        print(f"Removing existing directory: {yolo_data_dir}")
        shutil.rmtree(yolo_data_dir)

    # Prepare training and validation data
    train_ann_file = os.path.join(ProjectPaths.det_dataset, f'train_annotations_fold_{fold_k}.txt')
    val_ann_file = os.path.join(ProjectPaths.det_dataset, f'val_annotations_fold_{fold_k}.txt')
    
    prepare_fold_data(train_ann_file, yolo_data_dir, 'train', stream_mode)
    prepare_fold_data(val_ann_file, yolo_data_dir, 'val', stream_mode)

    # 2. Create YAML configuration
    yaml_content = f"""
train: {os.path.join(yolo_data_dir, 'train', 'images')}
val: {os.path.join(yolo_data_dir, 'val', 'images')}

nc: 1
names: ['mass']
"""
    yaml_path = os.path.join(yolo_data_dir, f'data_fold_{fold_k}.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Created dataset YAML at: {yaml_path}")

    # 3. Train YOLO
    model = YOLO(MODEL_VARIANT)
    
    print(f"Starting training for {EPOCHS} epochs. Freezing the first {FREEZE_LAYERS} layers.")
    total_layers = len([m for m in model.model.modules() if isinstance(m, torch.nn.Module)])
    trainable_layers = total_layers - FREEZE_LAYERS
    print(f"Total layers: {total_layers}")
    # print(f"Frozen layers: {FREEZE_LAYERS}")
    print(f"Trainable layers: {trainable_layers}")

    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=PROJECT_NAME,
        name=f"fold_{fold_k}_{stream_mode}",
        exist_ok=True,
        freeze=FREEZE_LAYERS,
        device=0 if torch.cuda.is_available() else 'cpu',
        augment= False
    )

    # 4. Extract Final mAP50
    results_path = os.path.join(results.save_dir, 'results.csv')
    try:
        df = pd.read_csv(results_path)
        map50_col_name = [col for col in df.columns if 'mAP50(B)' in col][0]
        final_map50 = df[map50_col_name].iloc[-1]
        print(f"--- Fold {fold_k} Finished. Final Validation mAP50: {final_map50:.4f} ---")
        return final_map50
    except Exception as e:
        print(f"Error reading results for fold {fold_k}: {e}")
        return 0.0

if __name__ == "__main__":
    all_fold_maps = []
    
    for k in range(NUM_FOLDS):
        fold_map = run_fold_training(fold_k=k, stream_mode=STREAM_MODE)
        all_fold_maps.append(fold_map)

    # Final Summary
    print(f"\n\n{'#'*30}\nCROSS-VALIDATION SUMMARY (Stream Mode: {STREAM_MODE})\n{'#'*30}")
    
    summary_df = pd.DataFrame({
        'Fold': [f'Fold {i}' for i in range(NUM_FOLDS)],
        'mAP50': all_fold_maps
    })
    
    mean_map = summary_df['mAP50'].mean()
    std_map = summary_df['mAP50'].std()
    
    summary_df.loc[NUM_FOLDS] = ['Average', mean_map]
    summary_df.loc[NUM_FOLDS + 1] = ['Std. Dev.', std_map]
    summary_df['mAP50'] = summary_df['mAP50'].apply(lambda x: f"{x:.4f}")

    print(summary_df.to_string(index=False))
    print(f"\nTraining logs and models saved in project directory: '{PROJECT_NAME}'")
import datetime
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .single_stream_dataloader import SingleStreamDataset, single_stream_collate
from .single_stream_retinanet import SingleStreamRetinaNet
from ..immutables import Hyperparameter, ProjectPaths
from ..callbacks import LossHistory
from ..utils import cvtColor, get_lr, preprocess_input, resize_image
from ..medical_image_utils import min_max_normalise
from .utils.utils_bbox import decodebox, non_max_suppression
from .utils.utils_map import get_map


def evaluate_map_on_validation_set(model, val_lines, cuda, save_dir):
    """Calculates mAP on the validation set for the current model state."""
    print("\nCalculating validation mAP...")
    model.eval()
    
    map_out_val = os.path.join(save_dir, "map_out_val_temp")
    if os.path.exists(map_out_val): shutil.rmtree(map_out_val)
    os.makedirs(map_out_val)
    os.makedirs(os.path.join(map_out_val, "ground-truth"))
    os.makedirs(os.path.join(map_out_val, "detection-results"))

    detection_class_name = "mass"
    
    # Determine which stream(s) to evaluate (default 'cm').
    stream_mode = getattr(Hyperparameter, "single_stream_mode", "cm").lower()

    # In single-stream, we evaluate each image from the val set independently
    val_images_to_process = []
    for line in val_lines:
        parts = line.strip().split()
        if not parts: continue
        orig = parts[0]
        if stream_mode == 'both':
            val_images_to_process.append(orig)
            val_images_to_process.append(orig.replace("_CM_", "_DM_") if "_CM_" in orig else orig.replace("_DM_", "_CM_"))
        elif stream_mode == 'cm':
            if "_CM_" in orig:
                val_images_to_process.append(orig)
            else:
                val_images_to_process.append(orig.replace("_DM_", "_CM_"))
        elif stream_mode == 'dm':
            if "_DM_" in orig:
                val_images_to_process.append(orig)
            else:
                val_images_to_process.append(orig.replace("_CM_", "_DM_"))
        else:
            val_images_to_process.append(orig)
            val_images_to_process.append(orig.replace("_CM_", "_DM_") if "_CM_" in orig else orig.replace("_DM_", "_CM_"))

    # Create a lookup for annotations
    annotation_lookup = {}
    for line in val_lines:
        parts = line.strip().split()
        base_name = os.path.splitext(parts[0])[0].replace("_CM_", "_").replace("_DM_", "_")
        annotation_lookup[base_name] = [list(map(int, bbox_str.split(',')))[:4] for bbox_str in parts[2:]]

    for image_name in tqdm(val_images_to_process, desc="Evaluating mAP on validation set"):
        image_id = Path(image_name).stem
        base_name_lookup = image_id.replace("_CM_", "_").replace("_DM_", "_")
        gt_boxes = annotation_lookup.get(base_name_lookup, [])

        image_path = os.path.join(ProjectPaths.det_dataset, image_name)

        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Warning: Could not open validation image {image_path}. Skipping.")
            continue
            
        image_height, image_width = np.array(np.shape(image)[0:2])
        # image_data = np.expand_dims(np.transpose(preprocess_input(np.array(resize_image(cvtColor(image), (Hyperparameter.input_shape[1], Hyperparameter.input_shape[0]), False), dtype='float32')), (2, 0, 1)), 0)
        image_data = np.expand_dims(np.transpose(min_max_normalise(np.array(resize_image(cvtColor(image), (Hyperparameter.input_shape[1], Hyperparameter.input_shape[0]), False), dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor).to(model.device)
            regression, classification, anchors = model(images, cuda=cuda)
            decoded_boxes = decodebox(regression[0], anchors[0], Hyperparameter.input_shape)
            prediction_tensor = torch.cat([decoded_boxes, classification[0]], axis=-1)
            results = non_max_suppression(prediction_tensor, Hyperparameter.input_shape, (image_height, image_width), False, conf_thres=0.02, nms_thres=0.5)

        with open(os.path.join(map_out_val, "ground-truth", f"{image_id}.txt"), "w") as f:
            for box in gt_boxes: f.write(f"{detection_class_name} {box[0]} {box[1]} {box[2]} {box[3]}\n")

        with open(os.path.join(map_out_val, "detection-results", f"{image_id}.txt"), "w") as f:
            if results[0] is not None:
                for i, box in enumerate(results[0][:, :4]):
                    f.write(f"{detection_class_name} {results[0][i, 4]} {int(box[1])} {int(box[0])} {int(box[3])} {int(box[2])}\n")

    get_map(MINOVERLAP=0.5, draw_plot=False, path=str(map_out_val))
    mass_ap = 0.0
    try:
        with open(os.path.join(map_out_val, "results/results.txt"), "r") as f:
            for line in f:
                if "mass AP" in line: mass_ap = float(line.split('%')[0]) / 100.0; break
    except Exception as e:
        print(f"Could not calculate or parse mAP for validation set. Error: {e}")
    
    shutil.rmtree(map_out_val)
    return mass_ap


def fit_one_epoch(model_train, model, optimizer, epoch, end_epoch, epoch_step, epoch_step_val, gen, gen_val, val_lines, Cuda, save_dir, stage_prefix, current_best_map):
    total_cls_loss, total_reg_loss, val_loss = 0, 0, 0
    device = model.device

    model_train.train()
    print(f"Start Train (lr: {get_lr(optimizer)[0]})")
    with tqdm(total=epoch_step, desc=f"Epoch {epoch + 1}/{end_epoch}", postfix=dict, mininterval=0.3) as pbar:
        for i, (images, bbox_targets) in enumerate(gen):
            if i >= epoch_step: break
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
                bbox_targets = [torch.from_numpy(ann).type(torch.FloatTensor).to(device) for ann in bbox_targets]

            optimizer.zero_grad()
            cls_loss, reg_loss = model_train(images, bbox_targets, cuda=Cuda)
            total_loss = cls_loss + reg_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
            optimizer.step()

            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            pbar.set_postfix(**{"cls": total_cls_loss / (i + 1), "reg": total_reg_loss / (i + 1)})
            pbar.update(1)

    model_train.eval()
    print("Start Validation")
    with tqdm(total=epoch_step_val, desc=f"Epoch {epoch + 1}/{end_epoch}", postfix=dict, mininterval=0.3) as pbar:
        for i, (images, bbox_targets) in enumerate(gen_val):
            if i >= epoch_step_val: break
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor).to(device)
                bbox_targets = [torch.from_numpy(ann).type(torch.FloatTensor).to(device) for ann in bbox_targets]
                cls_loss, reg_loss = model_train(images, bbox_targets, cuda=Cuda)
                val_loss += cls_loss.item() + reg_loss.item()
            pbar.set_postfix(**{"val_loss": val_loss / (i + 1)})
            pbar.update(1)

    val_map = evaluate_map_on_validation_set(model, val_lines, Cuda, save_dir)
    train_loss = (total_cls_loss + total_reg_loss) / epoch_step
    val_loss /= epoch_step_val
    loss_history.append_loss(epoch + 1, train_loss, val_loss, val_map)
    print(f"Epoch:{epoch + 1}/{end_epoch}, Total Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val mAP: {val_map:.3f}")

    file_suffix = f"ep{epoch + 1:03d}-loss{train_loss:.3f}-val_loss{val_loss:.3f}-val_map{val_map:.3f}.pth"
    for f in glob.glob(os.path.join(save_dir, f"{stage_prefix}last_*.pth")): os.remove(f)
    last_model_path = os.path.join(save_dir, f"{stage_prefix}last_{file_suffix}")
    torch.save(model.state_dict(), last_model_path)

    if val_map > current_best_map:
        print(f"New best model found (mAP {val_map:.3f} > {current_best_map:.3f}). Saving...")
        for f in glob.glob(os.path.join(save_dir, f"{stage_prefix}best_*.pth")): os.remove(f)
        best_model_path = os.path.join(save_dir, f"{stage_prefix}best_{file_suffix}")
        torch.save(model.state_dict(), best_model_path)
        current_best_map = val_map
        
    return last_model_path, current_best_map

def run_training_stage(stage_name, model, start_epoch, end_epoch, train_gen, val_gen, val_lines, epoch_step, epoch_step_val, base_lr, freeze_backbone, save_dir, initial_weights_path=None):
    print(f"\n{'='*20} STARTING {stage_name.upper()} {'='*20}")
    if initial_weights_path:
        print(f"Loading weights from: {initial_weights_path}")
        model.load_state_dict(torch.load(initial_weights_path, map_location=model.device))

    print("Applying freezing strategy for this stage...")
    for name, param in model.named_parameters():
        if name.startswith("backbone"):
            param.requires_grad = not freeze_backbone
    
    if freeze_backbone:
        if Hyperparameter.freeze_fpn_adaptation:
            print("Freezing FPN adaptation layers...")
            fpn_layers_to_freeze_single = [name.replace('le_fpn.', 'fpn.').replace('ce_fpn.', 'fpn.') for name in Hyperparameter.fpn_layers_to_freeze]
            for name, param in model.named_parameters():
                if name in fpn_layers_to_freeze_single: param.requires_grad = False
        if Hyperparameter.unfreeze_final_layers:
            print("Unfreezing final backbone layers...")
            layers_to_unfreeze_single = [name.replace('le_backbone.', 'backbone.').replace('ce_backbone.', 'backbone.') for name in Hyperparameter.layer_names_to_unfreeze]
            for name, param in model.named_parameters():
                if name in layers_to_unfreeze_single: param.requires_grad = True

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters for {stage_name}: {num_trainable:,}\n")

    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=Hyperparameter.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Hyperparameter.lr_scheduler_gamma)
    
    Cuda = torch.cuda.is_available()
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.to(model.device)

    best_map_for_stage = -1.0
    last_model_path = ""
    for epoch in range(start_epoch, end_epoch):
        last_model_path, best_map_for_stage = fit_one_epoch(
            model_train, model, optimizer, epoch, end_epoch,
            epoch_step, epoch_step_val, train_gen, val_gen, val_lines, Cuda,
            save_dir,
            stage_prefix=f"s{stage_name[6]}_",
            current_best_map=best_map_for_stage,
        )
        lr_scheduler.step()
    
    print(f"{'='*20} FINISHED {stage_name.upper()} {'='*20}\n")
    return last_model_path

def run_training_for_fold(fold_k):
    Cuda = torch.cuda.is_available()
    device = torch.device("cuda" if Cuda else "cpu")
    save_dir = os.path.join(ProjectPaths.det_training_logs, f"single_stream_fold_{fold_k}")
    os.makedirs(save_dir, exist_ok=True)
    
    train_ann_path = os.path.join(ProjectPaths.det_dataset, f'train_annotations_fold_{fold_k}.txt')
    val_ann_path = os.path.join(ProjectPaths.det_dataset, f'val_annotations_fold_{fold_k}.txt')
    
    with open(train_ann_path) as f: train_lines = f.readlines()
    with open(val_ann_path) as f: val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)
    
    if num_train == 0 or num_val == 0:
        print(f"Skipping Fold {fold_k} due to missing data (train: {num_train}, val: {num_val}).")
        return

    model = SingleStreamRetinaNet(num_classes=Hyperparameter.num_classes, phi=2)
    model.device = device
    if os.path.exists(ProjectPaths.pretrained_retinanet):
        model.load_pretrained_weights(ProjectPaths.pretrained_retinanet)
    
    global loss_history
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(save_dir, f"loss_{time_str}")
    loss_history = LossHistory(log_dir, model, input_shape=Hyperparameter.input_shape)

    # Which stream to include for this run (expects 'both'/'cm'/'dm' in Hyperparameter.single_stream_mode)
    include_stream_mode = getattr(Hyperparameter, "single_stream_mode", "cm")

    # --- STAGE 1 ---
    s1_batch_size = Hyperparameter.first_stage_batch_size
    train_dataset_s1 = SingleStreamDataset(train_lines, Hyperparameter.input_shape, Hyperparameter.num_classes, train=True, include_stream=include_stream_mode)
    val_dataset_s1 = SingleStreamDataset(val_lines, Hyperparameter.input_shape, Hyperparameter.num_classes, train=False, include_stream=include_stream_mode)
    gen_s1 = DataLoader(train_dataset_s1, shuffle=True, batch_size=s1_batch_size, num_workers=Hyperparameter.num_workers, pin_memory=True, drop_last=True, collate_fn=single_stream_collate)
    gen_val_s1 = DataLoader(val_dataset_s1, shuffle=False, batch_size=s1_batch_size, num_workers=Hyperparameter.num_workers, pin_memory=True, drop_last=True, collate_fn=single_stream_collate)
    epoch_step_s1, epoch_step_val_s1 = train_dataset_s1.length // s1_batch_size, val_dataset_s1.length // s1_batch_size

    last_model_s1 = run_training_stage("Stage 1", model, 0, 10, gen_s1, gen_val_s1, val_lines, epoch_step_s1, epoch_step_val_s1, Hyperparameter.first_stage_lr, True, save_dir)
    
    # --- STAGE 2 ---
    s2_batch_size = Hyperparameter.second_stage_batch_size
    train_dataset_s2 = SingleStreamDataset(train_lines, Hyperparameter.input_shape, Hyperparameter.num_classes, train=True, include_stream=include_stream_mode)
    val_dataset_s2 = SingleStreamDataset(val_lines, Hyperparameter.input_shape, Hyperparameter.num_classes, train=False, include_stream=include_stream_mode)
    gen_s2 = DataLoader(train_dataset_s2, shuffle=True, batch_size=s2_batch_size, num_workers=Hyperparameter.num_workers, pin_memory=True, drop_last=True, collate_fn=single_stream_collate)
    gen_val_s2 = DataLoader(val_dataset_s2, shuffle=False, batch_size=s2_batch_size, num_workers=Hyperparameter.num_workers, pin_memory=True, drop_last=True, collate_fn=single_stream_collate)
    epoch_step_s2, epoch_step_val_s2 = train_dataset_s2.length // s2_batch_size, val_dataset_s2.length // s2_batch_size

    run_training_stage("Stage 2", model, 10, 20, gen_s2, gen_val_s2, val_lines, epoch_step_s2, epoch_step_val_s2, Hyperparameter.second_stage_lr, False, save_dir, initial_weights_path=last_model_s1)

if __name__ == "__main__":
    NUM_FOLDS = 5
    for k in range(NUM_FOLDS):
        print(f"\n\n{'#'*30} FOLD {k + 1} / {NUM_FOLDS} {'#'*30}")
        run_training_for_fold(fold_k=k)
        print(f"{'#'*30} COMPLETED FOLD {k + 1} / {NUM_FOLDS} {'#'*30}")
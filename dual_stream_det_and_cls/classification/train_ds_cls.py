"""
"""

import datetime
import glob
import os
import re

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np


from .ds_dataloader_cls import DualStreamDataset, dual_stream_collate
from .ds_cls import DualStreamClassification
from ..immutables import Hyperparameter, ProjectPaths
from ..callbacks import LossHistory
from ..utils import get_lr

def evaluate_classification_on_validation_set(model, gen_val):
    """Calculates classification metrics on the validation set.
    
    Args:
        model: The DualStreamResNet model
        gen_val: Validation data generator
    
    Returns:
        dict: Dictionary containing metrics (f1, accuracy, precision, recall, confusion matrix values)
    """
    print("\nCalculating validation classification metrics...")
    model.eval()

    y_true = []
    y_pred = []
    
    classification_threshold = 0.5

    with torch.no_grad():
        for le_images, ce_images, labels in tqdm(gen_val, desc="Evaluating on validation set"):
            # Move inputs to device
            le_images = torch.from_numpy(le_images).type(torch.FloatTensor).to(model.device)
            ce_images = torch.from_numpy(ce_images).type(torch.FloatTensor).to(model.device)
            labels = torch.from_numpy(labels).type(torch.FloatTensor).to(model.device)
            
            # Forward pass with validation flag
            img_loss, predictions = model(le_images, ce_images, labels, is_validating=True)
            
            # Convert predictions to class labels based on model type
            if model.num_classes == 2:  # Binary classification
                preds = (predictions.cpu().numpy() > classification_threshold).astype(int).flatten()
            else:  # Multi-class classification
                preds = predictions.cpu().numpy().argmax(axis=1)
            
            # Store ground truth and predictions
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds)

    if not y_true:
        print("Warning: No samples in validation set for classification evaluation.")
        return {
            "f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0
        }

    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='binary' if model.num_classes == 2 else 'weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary' if model.num_classes == 2 else 'weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary' if model.num_classes == 2 else 'weighted', zero_division=0)
    
    try:
        if model.num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            cm = confusion_matrix(y_true, y_pred)
            # For multi-class, sum up the relevant confusion matrix values
            tp = np.sum(np.diag(cm))  # Sum of true positives for all classes
            fp = np.sum(cm.sum(axis=0) - np.diag(cm))  # Sum of false positives for all classes
            fn = np.sum(cm.sum(axis=1) - np.diag(cm))  # Sum of false negatives for all classes
            tn = np.sum(cm) - (tp + fp + fn)  # The rest are true negatives
    except ValueError:  # Happens if only one class is predicted
        tn, fp, fn, tp = 0, 0, 0, 0
        if all(p == 1 for p in y_pred) and all(t == 1 for t in y_true): 
            tp = len(y_true)
        elif all(p == 0 for p in y_pred) and all(t == 0 for t in y_true): 
            tn = len(y_true)

    metrics = {
        "f1": f1, "accuracy": accuracy, "precision": precision, "recall": recall,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }
    
    print("\n--- Validation Classification Metrics ---")
    print(f"  - F1-Score:  {f1:.4f}")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("---------------------------------------")
    
    return metrics


def apply_label_smoothing(img_class_targets, class_distribution, alpha, num_classes):
    """
    Apply label smoothing using class distribution (unigram frequencies)
    
    Args:
        img_class_targets: Original one-hot or binary targets
        class_distribution: Dictionary mapping class indices to their frequencies
        alpha: Label smoothing parameter (0 to 1)
        num_classes: Number of classes (2 or 3)
    
    Returns:
        Smoothed labels tensor
    """
    if num_classes not in [2, 3]:
        raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")

    if num_classes == 2:
        # Binary classification (Benign = 0, Malignant = 1)
        p_pos = class_distribution[1]  # Probability of Malignant
        
        # Create smoothed labels
        smoothed_pos_label = (1.0 - alpha) + alpha * p_pos
        smoothed_neg_label = alpha * p_pos
        
        # Apply smoothing
        smoothed_targets = torch.full_like(img_class_targets, smoothed_neg_label)
        smoothed_targets[img_class_targets == 1] = smoothed_pos_label
        
    else:  # num_classes == 3
        # Multi-class (Benign = 0, Malignant = 1, Normal = 2)
        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros((img_class_targets.size(0), 3), 
                                    device=img_class_targets.device)
        targets_one_hot.scatter_(1, img_class_targets.unsqueeze(1).long(), 1)
        
        # Calculate smoothed values for each class
        smoothed_targets = torch.zeros_like(targets_one_hot)
        for i in range(3):
            # For true class: (1-alpha) + alpha*p(class)
            # For other classes: alpha*p(class)
            p_class = class_distribution[i]
            mask = (img_class_targets == i)
            
            # True class
            smoothed_targets[mask, i] = (1.0 - alpha) + alpha * p_class
            
            # Other classes
            other_classes = [j for j in range(3) if j != i]
            for j in other_classes:
                smoothed_targets[mask, j] = alpha * class_distribution[j]
    
    return smoothed_targets


# --- Main Training Function ---
def fit_one_epoch(
    model_train,
    model,
    optimizer,
    epoch,
    epoch_step,
    gen,
    gen_val,
    End_Epoch,
    Cuda,
    save_dir,
    class_distribution,
    num_classes = 2
):
    total_img_loss = 0

    # Training
    model_train.train()
    print("Start Train")
    learning_rates = get_lr(optimizer)
    print(f"lr: {learning_rates[0]}")
    with tqdm(
        total=epoch_step,
        desc=f"Epoch {epoch + 1}/{End_Epoch}",
        postfix=dict,
        mininterval=0.3,
    ) as pbar:
        for i, batch in enumerate(gen):
            if i >= epoch_step:
                break
            le_images, ce_images, img_class_targets = batch

            with torch.no_grad():
                le_images = torch.from_numpy(le_images).type(torch.FloatTensor).to(model.device)
                ce_images = torch.from_numpy(ce_images).type(torch.FloatTensor).to(model.device)
                img_class_targets = torch.from_numpy(img_class_targets).type(torch.FloatTensor).to(model.device)

            # --- Unigram Label Smoothing ---
            if Hyperparameter.use_unigram_label_smoothing and model_train.training:
                img_class_targets = apply_label_smoothing(
                img_class_targets,
                class_distribution,
                Hyperparameter.label_smoothing_alpha,
                num_classes
            )

            optimizer.zero_grad()
            img_loss = model_train(le_images, ce_images, img_class_targets, is_validating=False)

            total_loss = img_loss 
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
            optimizer.step()

            total_img_loss += img_loss.item()
            
            pbar.set_postfix(
                **{
                    "img_cls": total_img_loss / (i + 1),
                }
            )
            pbar.update(1)

    # --- Validation and Evaluation ---
    metrics = evaluate_classification_on_validation_set(model, gen_val)
    val_f1 = metrics["f1"]
    
    train_loss = total_img_loss / epoch_step
    
    # Use F1-score for loss history instead of mAP for this stage
    loss_history.append_loss(epoch + 1, train_loss, val_f1, val_f1) # Using val_f1 for both val_loss and val_map slots for simplicity
    print(
        f"Epoch:{epoch + 1}/{End_Epoch}, Train img cls Loss: {train_loss:.3f}, Val F1-Score: {val_f1:.3f}"
    )

    # --- Save Models ---
    for f in glob.glob(os.path.join(save_dir, "cls_last_*.pth")): os.remove(f)
    last_model_name = f"cls_last_ep{epoch + 1:03d}-loss{train_loss:.3f}-f1{val_f1:.3f}.pth"
    torch.save(model.state_dict(), os.path.join(save_dir, last_model_name))

    # Save the best model based on F1-Score
    best_model_name = f"cls_best_ep{epoch + 1:03d}-loss{train_loss:.3f}-f1{val_f1:.3f}.pth"
    best_model_path = os.path.join(save_dir, best_model_name)
    existing_best_files = glob.glob(os.path.join(save_dir, "cls_best_*.pth"))
    
    if not existing_best_files:
        print(f"Saving new best model (F1: {val_f1:.3f}): {best_model_name}")
        torch.save(model.state_dict(), best_model_path)
    else:
        old_best_file = existing_best_files[0]
        last_best_f1 = -1.0
        try:
            match = re.search(r"f1([\d.]+)\.pth", old_best_file)
            if match:
                last_best_f1 = float(match.group(1))
        except (IndexError, ValueError):
            print(f"Warning: Could not parse F1-score from '{os.path.basename(old_best_file)}'.")

        if val_f1 > last_best_f1:
            print(f"New best model found (F1 {val_f1:.3f} > {last_best_f1:.3f}).")
            os.remove(old_best_file)
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model: {best_model_name}")


if __name__ == "__main__":
    Cuda = torch.cuda.is_available()
    device = torch.device("cuda" if Cuda else "cpu")

    # --- STAGE 3 CONFIG ---
    # Load the best model from the detection training stage
    MODEL_PATH = ProjectPaths.best_and_last_cls_model + "/cls_best_ep001-loss0.675-f10.837.pth"
    # MODEL_PATH = ''
    
    # --- MODEL SETUP ---
    num_classes = num_classes = 3 if Hyperparameter.include_normal_class else 2
    model = DualStreamClassification(num_classes, Hyperparameter.backbone)
    model.device = device
    
    # Load pre-trained ResNet-50 backbone weights
    # if os.path.exists(ProjectPaths.pretrained_cls):
    #     model.load_pretrained_weights(ProjectPaths.pretrained_cls)
    # else:
    #     print("Pre-trained weights not found, starting from scratch.")

    # Resume training from a checkpoint if specified
    if os.path.exists(MODEL_PATH):
        print(f"Loading weights from checkpoint {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Successfully loaded checkpoint weights.")
 

    # --- FREEZING STRATEGY ---
    if Hyperparameter.freeze_backbone_cls:
        print("\nFreezing backbone except image classification head and its fusion module...")
        for name, param in model.named_parameters():
            if name.startswith("le_backbone.") or name.startswith("ce_backbone"):
                param.requires_grad = False

    
    model_train = model.train()
    print(model) # Verify which layers are frozen/unfrozen
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.to(device)

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    log_dir = os.path.join(ProjectPaths.cls_training_logs, "cls_loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=Hyperparameter.input_shape, metric_name="F1-Score")

    # --- DATA LOADING ---
    with open(ProjectPaths.train_annotations_cls) as f:
        if Hyperparameter.include_normal_class:
            train_lines = f.readlines()
        else:
            train_lines = [line for line in f.readlines() if int(line.strip().split()[1]) in [0, 1]]
    with open(ProjectPaths.val_annotations_cls) as f:
        if Hyperparameter.include_normal_class:
            val_lines = f.readlines()
        else:
            val_lines = [line for line in f.readlines() if int(line.strip().split()[1]) in [0, 1]]
    num_train, num_val = len(train_lines), len(val_lines)
    
    class_distribution = None
    if Hyperparameter.use_unigram_label_smoothing:
        print("\nCalculating class distribution for label smoothing...")
        class_counts = {i: 0 for i in range(num_classes)}
        total_samples = 0
        
        for line in train_lines:
            label = int(line.strip().split()[1])
            class_counts[label] += 1
            total_samples += 1
            
        class_distribution = {
            cls: count/total_samples 
            for cls, count in class_counts.items()
        }
        
        print("Class distribution:")
        class_names = {0: "Benign", 1: "Malignant", 2: "Normal"}
        for cls in sorted(class_distribution.keys()):
            print(f"  - P({class_names[cls]:<8}) = {class_distribution[cls]:.4f}")


    if num_train > 0 and num_val > 0:
        # --- OPTIMIZER SETUP ---
        # Only pass trainable parameters to the optimizer
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, Hyperparameter.img_cls_lr, weight_decay=Hyperparameter.weight_decay_cls)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Hyperparameter.lr_scheduler_gamma_cls)
        
        # --- DATALOADERS ---
        train_dataset = DualStreamDataset(train_lines, Hyperparameter.input_shape, num_classes, train=True)
        val_dataset = DualStreamDataset(val_lines, Hyperparameter.input_shape, num_classes, train=False)
        gen = DataLoader(
            train_dataset, shuffle=True, batch_size=Hyperparameter.batch_size_cls, num_workers=Hyperparameter.num_workers,
            pin_memory=True, drop_last=True, collate_fn=dual_stream_collate
        )
        gen_val = DataLoader(
            val_dataset, shuffle=False, batch_size=Hyperparameter.batch_size_cls, num_workers=Hyperparameter.num_workers,
            pin_memory=True, drop_last=True, collate_fn=dual_stream_collate
        )

        epoch_step = num_train // Hyperparameter.batch_size_cls

        # --- TRAINING LOOP ---
        for epoch in range(Hyperparameter.init_epoch_cls, Hyperparameter.end_epoch_cls):
            fit_one_epoch(
                model_train,
                model,
                optimizer,
                epoch,
                epoch_step,
                gen,
                gen_val,
                Hyperparameter.end_epoch_cls,
                Cuda,
                ProjectPaths.best_and_last_cls_model,
                class_distribution,
                num_classes
            )
            lr_scheduler.step()
    else:
        print("Training did not start due to missing training or validation data.")
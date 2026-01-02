import datetime
import glob
import os
import re
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import specific project modules
from .ds_dataloader_cls import DualStreamDataset, dual_stream_collate
from .ds_cls import DualStreamClassification
from ..immutables import Hyperparameter, ProjectPaths
from ..callbacks import LossHistory
from ..utils import get_lr
from .train_ds_cls import evaluate_classification_on_validation_set, apply_label_smoothing

def fit_one_epoch_fold(
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
    loss_history,  # Passed explicitly
    num_classes=2
):
    """
    Train for one epoch and evaluate. 
    Adapted to accept loss_history as an argument for Cross-Validation.
    """
    total_img_loss = 0

    # Training
    model_train.train()
    learning_rates = get_lr(optimizer)
    
    pbar = tqdm(total=epoch_step, desc=f"Epoch {epoch + 1}/{End_Epoch}", mininterval=0.3)
    
    for i, batch in enumerate(gen):
        if i >= epoch_step:
            break
        le_images, ce_images, img_class_targets = batch

        with torch.no_grad():
            if Cuda:
                le_images = torch.from_numpy(le_images).type(torch.FloatTensor).cuda()
                ce_images = torch.from_numpy(ce_images).type(torch.FloatTensor).cuda()
                img_class_targets = torch.from_numpy(img_class_targets).type(torch.FloatTensor).cuda()
            else:
                le_images = torch.from_numpy(le_images).type(torch.FloatTensor)
                ce_images = torch.from_numpy(ce_images).type(torch.FloatTensor)
                img_class_targets = torch.from_numpy(img_class_targets).type(torch.FloatTensor)

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
        
        pbar.set_postfix(**{"img_cls": total_img_loss / (i + 1), "lr": learning_rates[0]})
        pbar.update(1)
    
    pbar.close()

    # --- Validation and Evaluation ---
    # evaluate_classification_on_validation_set is imported from train_ds_cls
    metrics = evaluate_classification_on_validation_set(model, gen_val)
    val_f1 = metrics["f1"]
    
    train_loss = total_img_loss / epoch_step
    
    # Append to the fold-specific loss history
    loss_history.append_loss(epoch + 1, train_loss, val_f1, val_f1)
    print(f"Epoch:{epoch + 1}/{End_Epoch}, Train Loss: {train_loss:.3f}, Val F1: {val_f1:.3f}")

    # --- Save Models ---
    # Remove previous 'last' model to save space
    for f in glob.glob(os.path.join(save_dir, "cls_last_*.pth")):
        os.remove(f)
    
    last_model_name = f"cls_last_ep{epoch + 1:03d}-loss{train_loss:.3f}-f1{val_f1:.3f}.pth"
    torch.save(model.state_dict(), os.path.join(save_dir, last_model_name))

    # Save Best Model logic
    existing_best_files = glob.glob(os.path.join(save_dir, "cls_best_*.pth"))
    best_model_name = f"cls_best_ep{epoch + 1:03d}-loss{train_loss:.3f}-f1{val_f1:.3f}.pth"
    
    if not existing_best_files:
        torch.save(model.state_dict(), os.path.join(save_dir, best_model_name))
        print(f"Saved new best model: {best_model_name}")
    else:
        old_best_file = existing_best_files[0]
        # Extract previous F1 from filename
        last_best_f1 = -1.0
        try:
            match = re.search(r"f1([\d.]+)\.pth", old_best_file)
            if match:
                last_best_f1 = float(match.group(1))
        except:
            pass

        if val_f1 > last_best_f1:
            print(f"New best model found (F1 {val_f1:.3f} > {last_best_f1:.3f}).")
            os.remove(old_best_file)
            torch.save(model.state_dict(), os.path.join(save_dir, best_model_name))

def run_fold(fold_idx, device, num_classes):
    print(f"\n{'='*20} Starting Fold {fold_idx} {'='*20}")
    
    # 1. Define paths (without creating directory yet)
    train_ann_path = os.path.join(ProjectPaths.cls_dataset, f"train_annotations_fold_{fold_idx}.txt")
    val_ann_path = os.path.join(ProjectPaths.cls_dataset, f"val_annotations_fold_{fold_idx}.txt")
    
    # Check if data exists
    if not os.path.exists(train_ann_path):
        print(f"Data for fold {fold_idx} not found. Skipping.")
        return

    # Define logging directory. 
    # NOTE: We DO NOT call os.makedirs here because LossHistory() in callbacks.py 
    # creates the directory and raises FileExistsError if it already exists.
    # We use a shared timestamp from the main block to group folds.
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    fold_save_dir = os.path.join(ProjectPaths.cls_training_logs, f"5fold_run", f"fold_{fold_idx}")
    
    # If the directory exists (e.g. from a previous crashed run), LossHistory might fail or append.
    # To be safe, if we are starting fresh, we rely on LossHistory. 
    # If LossHistory crashes on existing dir, we must handle it, but standard os.makedirs without exist_ok throws error.
    if os.path.exists(fold_save_dir):
        print(f"Warning: Log directory {fold_save_dir} already exists.")
        # If your LossHistory class crashes on existing dirs, you might want to rm -rf it here or append a timestamp.
        # Ideally, fix callbacks.py, but for now we append timestamp to ensure uniqueness.
        fold_save_dir = os.path.join(ProjectPaths.cls_training_logs, f"5fold_{time_str}", f"fold_{fold_idx}")

    # 2. Data Loading
    with open(train_ann_path) as f:
        train_lines = f.readlines()
    with open(val_ann_path) as f:
        val_lines = f.readlines()
        
    if not Hyperparameter.include_normal_class:
        train_lines = [x for x in train_lines if int(x.strip().split()[1]) in [0, 1]]
        val_lines = [x for x in val_lines if int(x.strip().split()[1]) in [0, 1]]

    num_train = len(train_lines)
    num_val = len(val_lines)
    print(f"Fold {fold_idx} Data: {num_train} training samples, {num_val} validation samples.")

    if num_train == 0 or num_val == 0:
        print("Skipping due to empty data.")
        return

    # 3. Model Initialization
    print("Initializing fresh model...")
    model = DualStreamClassification(num_classes, Hyperparameter.backbone)
    model.device = device
    
    if Hyperparameter.freeze_backbone_cls:
        for name, param in model.named_parameters():
            if name.startswith("le_backbone.") or name.startswith("ce_backbone"):
                param.requires_grad = False

    model_train = model.train()
    if torch.cuda.is_available():
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.to(device)

    # 4. Optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params, 
        Hyperparameter.img_cls_lr, 
        weight_decay=Hyperparameter.weight_decay_cls
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=Hyperparameter.lr_scheduler_gamma_cls
    )

    # 5. Class Distribution
    class_distribution = None
    if Hyperparameter.use_unigram_label_smoothing:
        class_counts = {i: 0 for i in range(num_classes)}
        total = 0
        for line in train_lines:
            lbl = int(line.strip().split()[1])
            class_counts[lbl] += 1
            total += 1
        class_distribution = {cls: count/total for cls, count in class_counts.items()}

    # 6. Datasets
    train_dataset = DualStreamDataset(train_lines, Hyperparameter.input_shape, num_classes, train=True)
    val_dataset = DualStreamDataset(val_lines, Hyperparameter.input_shape, num_classes, train=False)
    
    gen = DataLoader(
        train_dataset, shuffle=True, batch_size=Hyperparameter.batch_size_cls, 
        num_workers=Hyperparameter.num_workers, pin_memory=True, drop_last=True, 
        collate_fn=dual_stream_collate
    )
    gen_val = DataLoader(
        val_dataset, shuffle=False, batch_size=Hyperparameter.batch_size_cls, 
        num_workers=Hyperparameter.num_workers, pin_memory=True, drop_last=True, 
        collate_fn=dual_stream_collate
    )

    epoch_step = num_train // Hyperparameter.batch_size_cls
    
    # 7. Initialize LossHistory (This will create the directory)
    print(f"Logs will be saved to: {fold_save_dir}")
    try:
        loss_history = LossHistory(fold_save_dir, model, input_shape=Hyperparameter.input_shape, metric_name="F1-Score")
    except FileExistsError:
        print(f"Directory {fold_save_dir} exists. Attempting to use existing directory (LossHistory might fail if strict).")
        # If your callbacks.py is strict, this might fail again. Ideally, ensure unique path.
        # But since we generated a unique timestamp above, this shouldn't happen unless re-running quickly.
        pass

    # 8. Training Loop
    for epoch in range(Hyperparameter.init_epoch_cls, Hyperparameter.end_epoch_cls):
        fit_one_epoch_fold(
            model_train,
            model,
            optimizer,
            epoch,
            epoch_step,
            gen,
            gen_val,
            Hyperparameter.end_epoch_cls,
            torch.cuda.is_available(),
            fold_save_dir,
            class_distribution,
            loss_history, # Pass the fold-specific history
            num_classes
        )
        lr_scheduler.step()
        
        
    del model
    del model_train
    torch.cuda.empty_cache()
    print(f"Finished Fold {fold_idx}.\n")

if __name__ == "__main__":
    Cuda = torch.cuda.is_available()
    device = torch.device("cuda" if Cuda else "cpu")
    num_classes = 3 if Hyperparameter.include_normal_class else 2
    
    # Run 5 folds
    for i in range(5):
        run_fold(i, device, num_classes)
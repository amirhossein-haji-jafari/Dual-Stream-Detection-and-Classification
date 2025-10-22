import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    roc_curve
)

from .ds_dataloader_cls import DualStreamDataset, dual_stream_collate
from .ds_resnet import DualStreamClassification
from ..immutables import Hyperparameter, ProjectPaths

def test_model(model, test_loader, device, num_classes):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to run the model on (cuda or cpu).
        num_classes (int): The number of classes (2 for binary, 3 for multi-class).

    Returns:
        tuple: A tuple containing:
            - y_true (np.array): Ground truth labels.
            - y_pred (np.array): Predicted labels based on a 0.5 threshold.
            - y_scores (np.array): Raw prediction probabilities from the model.
            - image_names (list): List of image file names.
    """
    model.eval()
    y_true = []
    y_scores = []
    image_names = []
    
    print("Running inference on the test set...")
    with torch.no_grad():
        for le_images, ce_images, labels, names in tqdm(test_loader, desc="Testing"):
            le_images = torch.from_numpy(le_images).type(torch.FloatTensor).to(device)
            ce_images = torch.from_numpy(ce_images).type(torch.FloatTensor).to(device)
            
            # Forward pass to get probabilities
            predictions = model(le_images, ce_images) # In testing mode
            
            # For binary classification, scores are the probabilities of the positive class
            scores = predictions.cpu().numpy().flatten()

            y_scores.extend(scores)
            y_true.extend(labels)
            image_names.extend(names)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Convert probabilities to binary predictions
    classification_threshold = 0.5
    y_pred = (y_scores > classification_threshold).astype(int)
    
    return y_true, y_pred, y_scores, image_names

def calculate_and_save_metrics(y_true, y_pred, y_scores, image_names, output_dir):
    """
    Calculates and saves all required evaluation metrics and an ROC curve plot.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        y_scores (np.array): Raw prediction probabilities.
        image_names (list): List of image file names.
        output_dir (str): Directory to save the metrics file and ROC plot.
    """
    print("Calculating metrics...")
    # --- Confusion Matrix and its components ---
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # --- Core Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    
    # --- Predictive Values ---
    ppv = precision  # Positive Predictive Value is the same as Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # --- Save Metrics to File ---
    metrics_path = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("--- Model Evaluation Metrics ---\n\n")
        
        f.write("1. Confusion Matrix\n")
        f.write("----------------------------------------\n")
        f.write(f"                 | Predicted Benign | Predicted Malignant\n")
        f.write(f"-----------------|------------------|--------------------\n")
        f.write(f"Actual Benign    | TN = {tn:<12} | FP = {fp:<14}\n")
        f.write(f"Actual Malignant | FN = {fn:<12} | TP = {tp:<14}\n")
        f.write("----------------------------------------\n\n")

        f.write("2. Key Components\n")
        f.write(f"   - True Positives (TP):  {tp} (Malignant correctly identified)\n")
        f.write(f"   - True Negatives (TN):  {tn} (Benign correctly identified)\n")
        f.write(f"   - False Positives (FP): {fp} (Benign classified as Malignant)\n")
        f.write(f"   - False Negatives (FN): {fn} (Malignant classified as Benign)\n\n")
        
        f.write("3. Performance Metrics\n")
        f.write(f"   - Accuracy:           {accuracy:.4f}\n")
        f.write(f"   - Precision (PPV):    {precision:.4f}\n")
        f.write(f"   - Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"   - F1-Score:           {f1:.4f}\n")
        f.write(f"   - AUC:                {auc:.4f}\n")
        f.write(f"   - NPV:                {npv:.4f}\n\n")

        f.write("4. Misclassified Images\n")
        f.write("----------------------------------------\n")
        misclassified_indices = np.where(y_true != y_pred)[0]
        if len(misclassified_indices) > 0:
            for i in misclassified_indices:
                true_label = "Malignant" if y_true[i] == 1 else "Benign"
                pred_label = "Malignant" if y_pred[i] == 1 else "Benign"
                f.write(f"- {image_names[i]}: Actual ({true_label}), Predicted ({pred_label})\n")
        else:
            f.write("No misclassified images found.\n")
        
    print(f"Metrics saved to {metrics_path}")
    
    # --- Generate and Save ROC Curve Plot ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_plot_path = os.path.join(output_dir, "roc_curve.png")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(roc_plot_path)
    plt.close()
    
    print(f"ROC curve plot saved to {roc_plot_path}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Use the best model from your training logs
    MODEL_PATH = os.path.join(ProjectPaths.best_and_last_cls_model, "cls_best_ep012-loss0.329-f10.867.pth")
    TEST_ANNOTATIONS_PATH = ProjectPaths.val_annotations_cls 
    OUTPUT_DIR = ProjectPaths.dual_stream_cls + "/prediction_results"
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- SETUP ---
    Cuda = torch.cuda.is_available()
    device = torch.device("cuda" if Cuda else "cpu")
    
    num_classes = 2 # Assuming binary classification (Benign/Malignant)
    
    # --- LOAD MODEL ---
    print(f"Loading model from {MODEL_PATH}")
    model = DualStreamClassification(num_classes=num_classes, backbone=Hyperparameter.backbone)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print("Model loaded successfully.")
    
    # --- LOAD DATA ---
    with open(TEST_ANNOTATIONS_PATH) as f:
        # Filter out the 'Normal' class if it exists, since we focus on Benign vs. Malignant
        test_lines = [line for line in f.readlines() if int(line.strip().split()[1]) in [0, 1]]
    
    test_dataset = DualStreamDataset(test_lines, Hyperparameter.input_shape, num_classes, train=False, return_names=True)
    test_loader = DataLoader(
        test_dataset,
        shuffle=False, 
        batch_size=Hyperparameter.batch_size_cls, 
        num_workers=Hyperparameter.num_workers,
        pin_memory=True, 
        drop_last=False, 
        collate_fn=dual_stream_collate
    )

    # --- RUN EVALUATION ---
    if len(test_dataset) > 0:
        y_true, y_pred, y_scores, image_names = test_model(model, test_loader, device, num_classes)
        calculate_and_save_metrics(y_true, y_pred, y_scores, image_names, OUTPUT_DIR)
    else:
        print("Test dataset is empty. No evaluation was performed.")
import os
from dataclasses import dataclass, fields

PROJECT_ROOT = "/home/monstalinux/final-project/"


@dataclass(frozen=True)
class ProjectPaths:
    """
    Stores immutable file paths for the final project.

    This class uses a frozen dataclass to maintain consistent paths across the project,
    preventing accidental path modifications during runtime.

    Attributes:
        dual_stream (str): Path to the dual stream detection and classification project directory.
        registration_logs (str): Path to the image registration logs file.
        visualize_registration (str): Path to the directory for visualizing image registration results.
        dual_stream_det (str): Path to the dual stream detection module directory.
        dual_stream_cls (str): Path to the dual stream classification module directory.
        medical_reports (str): Path to the raw medical reports directory. this directory contains .docx files.
        manual_annotations (str): Path to the Excel file containing radiology manual annotations
        segmentations (str): Path to the CSV file containing radiology hand-drawn segmentations (mask).
        parsed_reports (str): Path to the CSV file containing processed report data for all cases.
        low_energy_images (str): Path to the directory containing low energy images of CDD-CESM
        low_energy_images_aligned (str): Path to the directory containing aligned low energy images of CDD-CESM
        subtracted_images (str): Path to the directory containing subtracted images of CDD-CESM
        subtracted_images_aligned (str): Path to the directory containing aligned subtracted images of CDD-CESM
        annotations_all_sheet_modified (str): Path to the Excel file containing modified annotations 'all' sheet.
        annotations_consistent_harmonized (str): Path to the CSV file containing consistent and harmonized annotations (for classification).
        font (str): Path to the font file used for rendering text in the application.
        det_definitions (str): Path to the .py file containing dataset definitions for detection. contains lists and dictionaries.
        pretrained_retinanet (str): Path to the pre-trained Standard RetinaNet model weights.
        det_training_logs (str): Path to the directory containing detection training logs.
        best_and_last_det_model (str): Path to the directory containing the best and last model weights.
        det_dataset_org (str): Path to the original detection dataset directory containing aligned images and annotations.
        det_annotations_org (str): Path to the text file containing original detection annotations.
        yolo_dataset (str): Path to the YOLO formatted dataset directory for detection.
        det_dataset (str): Path to the detection dataset directory containing augmented images and annotations.
        det_annotations (str): Path to the text file containing augmented detection annotations.
        train_annotations_fold_0 (str): Path to the text file containing training annotations of fold 0 for the DualStreamRetinaNet model.
        train_annotations_fold_1 (str): Path to the text file containing training annotations of fold 1 for the DualStreamRetinaNet model.
        train_annotations_fold_2 (str): Path to the text file containing training annotations of fold 2 for the DualStreamRetinaNet model.
        train_annotations_fold_3 (str): Path to the text file containing training annotations of fold 3 for the DualStreamRetinaNet model.
        train_annotations_fold_4 (str): Path to the text file containing training annotations of fold 4 for the DualStreamRetinaNet model.
        val_annotations_fold_0 (str): Path to the text file containing validation annotations of fold 0 for the DualStreamRetinaNet model.
        val_annotations_fold_1 (str): Path to the text file containing validation annotations of fold 1 for the DualStreamRetinaNet model.
        val_annotations_fold_2 (str): Path to the text file containing validation annotations of fold 2 for the DualStreamRetinaNet model.
        val_annotations_fold_3 (str): Path to the text file containing validation annotations of fold 3 for the DualStreamRetinaNet model.
        val_annotations_fold_4 (str): Path to the text file containing validation annotations of fold 4 for the DualStreamRetinaNet model.
        predictions (str): Path to the directory containing detection predictions.
        visualize_det_datasets (str): Path to the directory for visualizing detection dataset (original images and augmented).

        cls_training_logs (str): Path to the directory containing classification training logs.
        best_and_last_cls_model (str): Path to the directory containing the best and last classification model weights.
        cls_dataset_org (str): Path to the original classification dataset directory train and val sets.
        cls_dataset (str): Path to the classification dataset directory containing augmented images and annotations.
        cls_annotations (str): Path to the text file containing augmented classification annotations.
        train_annotations_cls (str): Path to the text file containing training annotations for classification.
        val_annotations_cls (str): Path to the text file containing validation annotations for classification.
        pretrained_cls (str): Path to the pre-trained classification model weights.
    """
    dual_stream: str = PROJECT_ROOT + "dual_stream_det_and_cls"
    registration_logs: str = dual_stream + "/registration_logs.txt"
    visualize_registration: str = dual_stream + "/visualize_registration"
    dual_stream_det: str = dual_stream + "/detection"
    dual_stream_cls: str = dual_stream + "/classification"

    medical_reports: str = PROJECT_ROOT + "dataset/Medical reports for cases"
    manual_annotations: str = PROJECT_ROOT + "dataset/Radiology_manual_annotations.xlsx"
    segmentations: str = PROJECT_ROOT + "dataset/Radiology_hand_drawn_segmentations_v2.csv"
    parsed_reports: str = PROJECT_ROOT + "dataset/Medical reports for cases.csv"
    low_energy_images: str = PROJECT_ROOT + "dataset/Low energy images of CDD-CESM"
    low_energy_images_aligned: str = PROJECT_ROOT + "dataset/Low energy images of CDD-CESM aligned"
    subtracted_images: str = PROJECT_ROOT + "dataset/Subtracted images of CDD-CESM"
    subtracted_images_aligned: str = PROJECT_ROOT + "dataset/Subtracted images of CDD-CESM aligned"
    annotations_all_sheet_modified: str = PROJECT_ROOT + "dataset/Radiology_manual_annotations_all_sheet_modified.csv"
    annotations_consistent_harmonized: str = PROJECT_ROOT + "dataset/annotations_consistent_harmonized.csv"
    font: str = dual_stream + "/CaskaydiaCoveNerdFontMono-Regular.ttf"
    
    # detection
    det_definitions: str = dual_stream_det + "/detection_data_preparation/detection_dataset_definitions.py"
    pretrained_retinanet: str = dual_stream_det + "/RetinaNet_ResNet50.pth"
    det_training_logs: str = dual_stream_det + "/detection_logs"
    best_and_last_det_model: str = det_training_logs + "/dual_stream"
    
    det_dataset_org: str = dual_stream_det + "/dataset_org"
    det_annotations_org: str = det_dataset_org + "/annotations.txt"

    yolo_dataset: str = dual_stream_det + "/yolo"

    det_dataset: str = dual_stream_det + '/dataset' # augmented dataset for detection
    det_annotations: str = det_dataset + "/augmented_annotations.txt"
    train_annotations_fold_0: str = det_dataset + "/train_annotations_fold_0.txt"
    train_annotations_fold_1: str = det_dataset + "/train_annotations_fold_1.txt"
    train_annotations_fold_2: str = det_dataset + "/train_annotations_fold_2.txt"
    train_annotations_fold_3: str = det_dataset + "/train_annotations_fold_3.txt"
    train_annotations_fold_4: str = det_dataset + "/train_annotations_fold_4.txt"
    val_annotations_fold_0: str = det_dataset + "/val_annotations_fold_0.txt"
    val_annotations_fold_1: str = det_dataset + "/val_annotations_fold_1.txt"
    val_annotations_fold_2: str = det_dataset + "/val_annotations_fold_2.txt"
    val_annotations_fold_3: str = det_dataset + "/val_annotations_fold_3.txt"
    val_annotations_fold_4: str = det_dataset + "/val_annotations_fold_4.txt"
    
    predictions: str = dual_stream_det + "/predictions"
    visualize_det_datasets: str = dual_stream_det + "/detection_dataset_evaluation/visualize"

    # classification
    cls_training_logs: str = dual_stream_cls + "/classification_logs"
    best_and_last_cls_model: str = cls_training_logs + "/dual_stream"
    cls_dataset_org: str = dual_stream_cls + "/dataset_org"
    cls_dataset: str = dual_stream_cls + "/dataset"
    cls_annotations: str = cls_dataset + "/augmented_cls_annotations.txt"
    train_annotations_cls: str = cls_dataset + "/train_annotations_augmented.txt"
    val_annotations_cls: str = cls_dataset + "/val_annotations.txt"
    pretrained_cls: str = dual_stream_cls + "/ianpan_mammoscreen.pth"


    def __str__(self):
        """Generates a string representation of the project's directory structure."""
        path_tree = {}

        # Sort fields by path to process parent directories before their children
        sorted_fields = sorted(fields(self), key=lambda f: getattr(self, f.name))

        for f in sorted_fields:
            path = getattr(self, f.name)
            norm_path = os.path.normpath(path)
            norm_root = os.path.normpath(PROJECT_ROOT)

            if not norm_path.startswith(norm_root):
                continue
            
            relative_path = os.path.relpath(norm_path, norm_root)
            if relative_path == '.':
                continue

            parts = relative_path.split(os.sep)
            current_level = path_tree

            for part in parts[:-1]:
                current_level = current_level.setdefault(part, {})
            
            leaf_name = parts[-1]
            
            # Heuristic: if a path has no file extension, treat it as a directory.
            # This allows us to correctly mark attributes pointing to directories.
            is_directory = os.path.splitext(leaf_name)[1] == ''

            if is_directory:
                node = current_level.setdefault(leaf_name, {})
                # Use a special key '.' to store the attribute name for the directory itself.
                node['.'] = f"({f.name})"
            else:
                current_level[leaf_name] = f"({f.name})"

        header = f"Project Structure (based on ProjectPaths):\n{PROJECT_ROOT}"
        tree_view = self._generate_tree_view(path_tree)
        return f"{header}\n{tree_view}"

    def _generate_tree_view(self, subtree, prefix=""):
        """Recursively builds the tree view string."""
        # Exclude our special directory marker from the items to print
        entries = sorted([item for item in subtree.items() if item[0] != '.'])
        lines = []

        for i, (name, content) in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if isinstance(content, dict):
                dir_attr = content.get('.', '')
                line_end = f" {dir_attr}" if dir_attr else ""
                lines.append(f"{prefix}{connector}{name}/{line_end}")
                
                # Recursive call for the subdirectory's content
                children_view = self._generate_tree_view(content, prefix + extension)
                if children_view:
                    lines.append(children_view)
            else:
                # This is a leaf node (a file)
                lines.append(f"{prefix}{connector}{name} {content}")

        return "\n".join(lines)

@dataclass(frozen=True)
class Hyperparameter:
    """
    Stores hyperparameters for the DualStreamRetinaNet and DualStreamClassification models.

    This class uses a frozen dataclass to maintain consistent hyperparameters across the project

    Attributes:
        input_shape (tuple[int, int]): Input shape for the model, typically [height, width]. 
            multiple of the RetinaNet's backbone total stride (which is often 128 for a ResNet-50/101 backbone)
        num_workers (int): Number of workers for data loading.
        init_epoch (int): Initial epoch for training.
        end_epoch (int): Final epoch for training. 
        first_stage_batch_size (int): Batch size for 1st stage of training of DualStreamRetinaNet.
            Adjust based on GPU memory, Input Shape and Trainable Parameters.
        second_stage_batch_size (int): Batch size for 2nd stage of training of DualStreamRetinaNet.
        single_stream_mode (str): Single stream mode for ablation studies. options are:
            'dm' : only use low-energy images.
            'cm' : only use subtracted images.
            'both' : use both low-energy and subtracted images.
        use_clahe (bool): Whether to apply CLAHE preprocessing to input images.
        first_stage_lr (float): learning rate for the 1st stage of training.
        second_stage_lr (float): learning rate for the 2nd stage of training.
        focal_loss_alpha (float): Alpha parameter for Focal Loss; balances importance of positive/negative examples.
        focal_loss_gamma (float): Gamma parameter for Focal Loss; focuses learning on hard examples
        smooth_l1_loss_beta (fload): Beta parameter for Smooth L1 loss; controls the transition point between L1 and L2 loss.
            Lower values make the loss closer to L1, higher values make it closer to L2.
        freeze_backbone (bool): Whether to freeze ResNet backbone network during training.
        freeze_fpn_adaptation (bool): Whether to freeze the FPN adaptation layers during training.
        fpn_layers_to_freeze (tuple[str, ...]): Tuple of FPN layer names to freeze during training.
            These layers are typically the first layers of each FPN level.
        unfreeze_final_layers (bool): Whether to unfreeze the final layers of ResNet backbone network.
        layer_names_to_unfreeze (tuple[str, ...]): Tuple of layer names to unfreeze during training.
        fusion_method (str): Which fusion method to use options are:
            gated
            max
            conv-relu
            relu-add
            conditional
            synergistic
        use_simple_reg_head (bool): Whether to use SimplifiedRegressionModel or RegressionModel.
        use_simple_cls_head (bool): Whether to use SimplifiedClassificationModel or ClassificationModel.
        num_classes (int): Number of classes for classification head of detection.
            will be 1 for objectness. 2 for 'benign' and 'malignant'.
        weight_decay (float): Weight decay (L2 Regularization) for the optimizer.
        lr_scheduler_gamma (float): Learning rate scheduler gamma value.
            Reduces the learning rate by this factor every step_size epochs.
        simple_reg_dropout_rate (float): Dropout rate for SimplifiedRegressionModel.
        reg_dropout_rate (float): Dropout rate for RegressionModel.
        simple_cls_dropout_rate (float): Dropout rate for SimplifiedClassificationModel.
        cls_dropout_rate (float): Dropout rate for ClassificationModel.
        fpn_p3_2_dropout_rate (float): Dropout rate for FPN P3_2 layer.
        fpn_p4_2_dropout_rate (float): Dropout rate for FPN P4_2 layer.
        fpn_p5_2_dropout_rate (float): Dropout rate for FPN P5_2 layer.
        fpn_p6_dropout_rate (float): Dropout rate for FPN P6 layer.
        fpn_p7_dropout_rate (float): Dropout rate for FPN P7 layer.

        fusion_method_cls (str): Which fusion method to use in classification head. options are:
            gated
            max
            conv-relu
            relu-add
            conditional
            synergistic
        img_cls_dropout_rate (float): Dropout rate for the image classification head.
        include_normal_class (bool): Whether to include 'normal' class in classification (2 classes if False, 3 if True).
        use_unigram_label_smoothing (bool): Whether to use unigram label smoothing for classification.
        label_smoothing_alpha (float): Alpha parameter for unigram label smoothing. defaults to 0.2.
        freeze_backbone_cls (bool): Whether to freeze the classification backbone during training.
        img_cls_lr (float): Learning rate for the classification model.
        weight_decay_cls (float): Weight decay (L2 Regularization) for the classification optimizer
        lr_scheduler_gamma_cls (float): Learning rate scheduler gamma value for classification model.
        batch_size_cls (int): Batch size for classification model training.
        init_epoch_cls (int): Initial epoch for classification model training.
        end_epoch_cls (int): Final epoch for classification model training.
        backbone (str): Backbone model for classification. options are:
            'resnet-of-retinanet' : uses the ResNet-50 backbone from RetinaNet.
            'ianpan_mammoscreen' : uses the ianpan/mammoscreen model from HuggingFace.

    """

    # Training parameters
    input_shape: tuple[int, int] = (640, 640)  # Input shape for the model, typically [height, width]
    num_workers: int = 8
    init_epoch: int = 0
    end_epoch: int = 10
    first_stage_batch_size: int = 16
    second_stage_batch_size: int = 4
    single_stream_mode: str = 'dm'
    use_clahe: bool = False

    # Learning rates
    first_stage_lr: float = 1e-3
    second_stage_lr: float = 2e-4

    # Loss function parameters
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 5.0
    smooth_l1_loss_beta: float = (1.0/9.0)

    # Freezing strategies
    freeze_backbone: bool = True
    freeze_fpn_adaptation: bool = True
    fpn_layers_to_freeze: tuple[str, ...] = (
        'le_fpn.P3_1.weight',
        'le_fpn.P3_1.bias',
        'le_fpn.P4_1.weight',
        'le_fpn.P4_1.bias',
        'le_fpn.P5_1.weight',
        'le_fpn.P5_1.bias',
        'le_fpn.P6.weight',
        'le_fpn.P6.bias',
        'ce_fpn.P3_1.weight',
        'ce_fpn.P3_1.bias',
        'ce_fpn.P4_1.weight',
        'ce_fpn.P4_1.bias',
        'ce_fpn.P5_1.weight',
        'ce_fpn.P5_1.bias',
        'ce_fpn.P6.weight',
        'ce_fpn.P6.bias',
    )

    # Unfreezing strategies
    unfreeze_final_layers: bool = False
    layer_names_to_unfreeze: tuple[str, ...] = (
            # "ce_backbone.model.layer4.2.conv2.weight",
            # "ce_backbone.model.layer4.2.bn2.weight",
            # "ce_backbone.model.layer4.2.bn2.bias",
            "ce_backbone.model.layer4.2.conv3.weight",
            "ce_backbone.model.layer4.2.bn3.weight",
            "ce_backbone.model.layer4.2.bn3.bias",
            # "le_backbone.model.layer4.2.conv2.weight",
            # "le_backbone.model.layer4.2.bn2.weight",
            # "le_backbone.model.layer4.2.bn2.bias",
            "le_backbone.model.layer4.2.conv3.weight",
            "le_backbone.model.layer4.2.bn3.weight",
            "le_backbone.model.layer4.2.bn3.bias",
    )
    
    # Model architecture
    fusion_method: str = "synergistic"
    use_simple_reg_head : bool = True
    use_simple_cls_head : bool = True
    num_classes: int = 1  
    weight_decay: float = 5e-4
    lr_scheduler_gamma: float = 0.85

    # Dropout rates
    simple_reg_dropout_rate: float = 0.10
    reg_dropout_rate = 0.20
    simple_cls_dropout_rate: float = 0.10
    cls_dropout_rate: float = 0.20

    # FPN dropout rates
    fpn_p3_2_dropout_rate: float = 0.10
    fpn_p4_2_dropout_rate: float = 0.10
    fpn_p5_2_dropout_rate: float = 0.10
    fpn_p6_dropout_rate: float = 0.20
    fpn_p7_dropout_rate: float = 0.10

    # Classification hyperparameters
    fusion_method_cls: str = "synergistic"
    img_cls_dropout_rate: float = 0.0
    include_normal_class: bool = False  # Whether to include 'normal' class in classification (2 classes if False, 3 if True)
    use_unigram_label_smoothing: bool = False 
    label_smoothing_alpha: float = 0.2
    freeze_backbone_cls: bool = False
    img_cls_lr: float = 1e-4 # start with 1e-3
    weight_decay_cls: float = 1e-4
    lr_scheduler_gamma_cls: float = 0.90
    batch_size_cls: int = 2
    init_epoch_cls: int = 0
    end_epoch_cls: int = 30
    backbone: str = "ianpan_mammoscreen" 


if __name__ == "__main__":
    print(ProjectPaths())
"""
Project Structure (based on ProjectPaths):
/home/monstalinux/final-project/
├── dataset/
│   ├── Low energy images of CDD-CESM/ (low_energy_images)
│   ├── Low energy images of CDD-CESM aligned/ (low_energy_images_aligned)
│   ├── Medical reports for cases/ (medical_reports)
│   ├── Medical reports for cases.csv (parsed_reports)
│   ├── Radiology_hand_drawn_segmentations_v2.csv (segmentations)
│   ├── Radiology_manual_annotations.xlsx (manual_annotations)
│   ├── Radiology_manual_annotations_all_sheet_modified.csv (annotations_all_sheet_modified)
│   ├── Subtracted images of CDD-CESM/ (subtracted_images)
│   ├── Subtracted images of CDD-CESM aligned/ (subtracted_images_aligned)
│   └── annotations_consistent_harmonized.csv (annotations_consistent_harmonized)
└── dual_stream_det_and_cls/ (dual_stream)
    ├── CaskaydiaCoveNerdFontMono-Regular.ttf (font)
    ├── classification/ (dual_stream_cls)
    │   ├── classification_logs/ (cls_training_logs)
    │   │   └── dual_stream/ (best_and_last_cls_model)
    │   ├── dataset/ (cls_dataset)
    │   │   ├── augmented_cls_annotations.txt (cls_annotations)
    │   │   ├── train_annotations_augmented.txt (train_annotations_cls)
    │   │   └── val_annotations.txt (val_annotations_cls)
    │   ├── dataset_org/ (cls_dataset_org)
    │   └── ianpan_mammoscreen.pth (pretrained_cls)
    ├── detection/ (dual_stream_det)
    │   ├── RetinaNet_ResNet50.pth (pretrained_retinanet)
    │   ├── dataset/ (det_dataset)
    │   │   ├── augmented_annotations.txt (det_annotations)
    │   │   ├── train_annotations_fold_0.txt (train_annotations_fold_0)
    │   │   ├── train_annotations_fold_1.txt (train_annotations_fold_1)
    │   │   ├── train_annotations_fold_2.txt (train_annotations_fold_2)
    │   │   ├── train_annotations_fold_3.txt (train_annotations_fold_3)
    │   │   ├── train_annotations_fold_4.txt (train_annotations_fold_4)
    │   │   ├── val_annotations_fold_0.txt (val_annotations_fold_0)
    │   │   ├── val_annotations_fold_1.txt (val_annotations_fold_1)
    │   │   ├── val_annotations_fold_2.txt (val_annotations_fold_2)
    │   │   ├── val_annotations_fold_3.txt (val_annotations_fold_3)
    │   │   └── val_annotations_fold_4.txt (val_annotations_fold_4)
    │   ├── dataset_org/ (det_dataset_org)
    │   │   └── annotations.txt (det_annotations_org)
    │   ├── detection_data_preparation/
    │   │   └── detection_dataset_definitions.py (det_definitions)
    │   ├── detection_dataset_evaluation/
    │   │   └── visualize/ (visualize_det_datasets)
    │   ├── detection_logs/ (det_training_logs)
    │   │   └── dual_stream/ (best_and_last_det_model)
    │   ├── predictions/ (predictions)
    │   └── yolo/ (yolo_dataset)
    ├── registration_logs.txt (registration_logs)
    └── visualize_registration/ (visualize_registration)
"""
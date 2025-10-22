"""
CDD-CESM Dataset contains paired grayscale mammography images, which are acquired using different modalities: one low-energy (DM) and one contrast-enhanced (or subtracted).
Both images of each pair are acquired from the exact same anatomical region and angle within a patient's breast.
Crucially, any mass visible in the low-energy image appears at an identical pixel location in the corresponding contrast-enhanced image, and vice versa.
The dataset includes precise bounding box annotations for masses present in both types of images and image-level “Malignant” or “Benign” labels.
Since the two images contain co-localized masses, each pair shares the same Bounding annotations for masses and image-level classification labels. Each Image could have more than one bounding box.
We want to perform object detection in a multi-modal context. We propose a modification of RetinaNet:
1.	Dual-Stream Backbone and FPN: uses two separate ResNet backbones and two Feature Pyramid Networks (FPNs), one for the low-energy (LE) image and one for the contrast-enhanced (CM) image.
Because These parts are architecturally identical a standard RetinaNet, We a manual mapping of pre-trained weights from the standard RetinaNet (as RetinaNet_ResNet50.pth) to This part of the net.
We load the same pre-trained weights into both of streams.
2.	Feature Fusion for Detection: The feature maps (P3-P7) from both FPNs are fused. The fusion strategy implementation is:
    Feature Fusion:
        A: Normalization: “LayerNorm” is applied to stabilize the features from each stream. LayerNorm is applied on the channel dimenssion for each pixel.
        B: Concatenation: The normalized feature maps are concatenated along the channel dimension.
        C: Convolution: A 1x1 convolution reduces the channel dimension back to the original, effectively mixing the information.
        D: ReLU activation: To add non-linearity to feature fusion.
    Simplified Feature Fusion:
        A: Normalization
        B: ReLU activation: Making sure features are not "alleviating" each other (also known as destructive interference)
            If a feature channel in x1_norm has a strong positive activation (e.g., +1.5) indicating a mass, 
            but the corresponding channel in x2_norm has a strong negative activation (e.g., -1.2), 
            the result will be +0.3. The strong signal from the first stream has been significantly weakened or "alleviated" by the second.
        C: Element wise addition 
3.	Modified Detection Head: 
A: The standard RetinaNet classification subnet predicts probabilities for `K` classes. We modify this to predict a single "objectness" score per anchor (i.e., is this a mass or background?).
This is because boxes don't have individual "Malignant/Benign" labels.
B: The regression subnet remains the same, predicting offsets for the bounding box.
C: The convolutional stack within detection head is identical to the standard RetinaNet. We transfer pre-trained weights. 
However, the final predictive layer of classification subnet is different (objectness vs. multi-class), so we skip that specific layer and let it keep its custom initialization.
The regression subnet is identical and can be fully loaded.
5.	Composite Loss: During training, the model calculates and combines two losses:
A: Objectness Loss: Focal Loss (standard for RetinaNet) to handle the imbalance between foreground (masses) and background anchors.
B: Regression Loss: Smooth L1 Loss for the bounding box coordinates.
"""

import torch
import torch.nn as nn

# necessary components from https://github.com/RollingHol/ERetinaNet/tree/master
from .retinanet import (Resnet, PyramidFeatures, RegressionModel, ClassificationModel,
                        SimplifiedRegressionModel, SimplifiedClassificationModel)
from .retinanet_training import FocalLoss
from .utils.anchors import Anchors
from ..immutables import Hyperparameter
from ..fusion import (ReluAddFusion, ConvReluFusion, GatedFusion, 
                      MaxFusion, ConditionalFusion, SynergisticFusion)

class DualStreamRetinaNet(nn.Module):
    def __init__(self, num_classes, phi=2, pretrained_backbone=None):
        super().__init__()
        # phi=2 corresponds to ResNet-50
        fpn_sizes = {2: [512, 1024, 2048]}[phi]

        # 1. Dual-Stream Backbone
        self.le_backbone = Resnet(phi, pretrained=pretrained_backbone)
        self.ce_backbone = Resnet(phi, pretrained=pretrained_backbone)
        
        # 2. Dual-Stream FPN
        self.le_fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], 
                                      p3_2_dropout_rate=Hyperparameter.fpn_p3_2_dropout_rate,
                                      p4_2_dropout_rate=Hyperparameter.fpn_p4_2_dropout_rate,
                                      p5_2_dropout_rate=Hyperparameter.fpn_p5_2_dropout_rate,
                                      p6_dropout_rate=Hyperparameter.fpn_p6_dropout_rate,
                                      p7_dropout_rate=Hyperparameter.fpn_p7_dropout_rate)
        self.ce_fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2],
                                      p3_2_dropout_rate=Hyperparameter.fpn_p3_2_dropout_rate,
                                      p4_2_dropout_rate=Hyperparameter.fpn_p4_2_dropout_rate,
                                      p5_2_dropout_rate=Hyperparameter.fpn_p5_2_dropout_rate,
                                      p6_dropout_rate=Hyperparameter.fpn_p6_dropout_rate,
                                      p7_dropout_rate=Hyperparameter.fpn_p7_dropout_rate)

        # 3. Feature Fusion for Detection (P3-P7)
        print(f"Using '{Hyperparameter.fusion_method}' feature fusion method.")
        match Hyperparameter.fusion_method:
            case 'relu-add':
                self.fusion_modules = nn.ModuleList([ReluAddFusion(256) for _ in range(5)])
            case 'conv-relu':
                self.fusion_modules = nn.ModuleList([ConvReluFusion(256) for _ in range(5)])
            case 'gated':
                self.fusion_modules = nn.ModuleList([GatedFusion(256) for _ in range(5)])
            case 'max':
                self.fusion_modules = nn.ModuleList([MaxFusion(256) for _ in range(5)])
            case 'conditional':
                self.fusion_modules = nn.ModuleList([ConditionalFusion(256) for _ in range(5)])
            case 'synergistic':
                self.fusion_modules = nn.ModuleList([SynergisticFusion(256) for _ in range(5)])
            case _:
                raise ValueError(
                    f"Fusion method is invalid. Expected 'relu-add', 'conv-relu', 'gated', 'max', 'conditional', or 'synergistic', "
                    f"but got '{Hyperparameter.fusion_method}'."
                )

        # 4. Modified Detection Head for "objectness"
        # classification head
        if Hyperparameter.use_simple_cls_head:
            self.classificationModel = SimplifiedClassificationModel(256, Hyperparameter.simple_cls_dropout_rate, num_classes=num_classes)
        else:
            self.classificationModel = ClassificationModel(256, Hyperparameter.cls_dropout_rate, num_classes=num_classes)
        
        # regression head
        if Hyperparameter.use_simple_reg_head:
            self.regressionModel = SimplifiedRegressionModel(256, Hyperparameter.simple_reg_dropout_rate)
        else:
            self.regressionModel = RegressionModel(256, Hyperparameter.reg_dropout_rate)

        # 6. Anchors and Loss functions
        self.anchors = Anchors()
        self.focal_loss = FocalLoss()
        # self.img_cls_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, le_image, ce_image, annotations=None, cuda=True):
        # --- Backbone pass ---
        c3_le, c4_le, c5_le = self.le_backbone(le_image)
        c3_ce, c4_ce, c5_ce = self.ce_backbone(ce_image)

        # FPN pass for both streams
        le_features_fpn = self.le_fpn([c3_le, c4_le, c5_le])
        ce_features_fpn = self.ce_fpn([c3_ce, c4_ce, c5_ce])
        
        # --- Detection Head ---
        features = [self.fusion_modules[i](le_features_fpn[i], ce_features_fpn[i]) for i in range(5)]
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(features)
        
        # --- Loss Calculation (Training/Validation) ---
        if annotations is not None:
            _, cls_loss, reg_loss = self.focal_loss(classification, regression, anchors, annotations, cuda=cuda,alpha=Hyperparameter.focal_loss_alpha, gamma=Hyperparameter.focal_loss_gamma)
        
            return cls_loss, reg_loss
        
        # --- Inference ---
        else:
            return regression, classification, anchors
            


    def load_pretrained_weights(self, pth_path):
        print(f"Loading pre-trained weights from {pth_path}")
        device = next(self.parameters()).device
        pretrained_dict = torch.load(pth_path, map_location=device)["state_dict"]
        model_dict = self.state_dict()
        mapped_weights = {}

        if Hyperparameter.use_simple_reg_head:
            print("Using SimplifiedRegressionModel; skipping weight mapping for regression head.")
        if Hyperparameter.use_simple_cls_head:
            print("Using SimplifiedClassificationModel; skipping weight mapping for classification head.")
            
        for key, value in pretrained_dict.items():
            if key.startswith('backbone.'):
                mapped_weights[key.replace('backbone.', 'le_backbone.model.', 1)] = value
                mapped_weights[key.replace('backbone.', 'ce_backbone.model.', 1)] = value
            elif key.startswith('neck.lateral_convs.0.conv.'):
                mapped_weights[key.replace('neck.lateral_convs.0.conv.', 'le_fpn.P3_1.', 1)] = value
                mapped_weights[key.replace('neck.lateral_convs.0.conv.', 'ce_fpn.P3_1.', 1)] = value
            elif key.startswith('neck.lateral_convs.1.conv.'):
                mapped_weights[key.replace('neck.lateral_convs.1.conv.', 'le_fpn.P4_1.', 1)] = value
                mapped_weights[key.replace('neck.lateral_convs.1.conv.', 'ce_fpn.P4_1.', 1)] = value
            elif key.startswith('neck.lateral_convs.2.conv.'):
                mapped_weights[key.replace('neck.lateral_convs.2.conv.', 'le_fpn.P5_1.', 1)] = value
                mapped_weights[key.replace('neck.lateral_convs.2.conv.', 'ce_fpn.P5_1.', 1)] = value
            elif key.startswith('neck.fpn_convs.0.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.0.conv.', 'le_fpn.P3_2.', 1)] = value
                mapped_weights[key.replace('neck.fpn_convs.0.conv.', 'ce_fpn.P3_2.', 1)] = value
            elif key.startswith('neck.fpn_convs.1.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.1.conv.', 'le_fpn.P4_2.', 1)] = value
                mapped_weights[key.replace('neck.fpn_convs.1.conv.', 'ce_fpn.P4_2.', 1)] = value
            elif key.startswith('neck.fpn_convs.2.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.2.conv.', 'le_fpn.P5_2.', 1)] = value
                mapped_weights[key.replace('neck.fpn_convs.2.conv.', 'ce_fpn.P5_2.', 1)] = value
            elif key.startswith('neck.fpn_convs.3.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.3.conv.', 'le_fpn.P6.', 1)] = value
                mapped_weights[key.replace('neck.fpn_convs.3.conv.', 'ce_fpn.P6.', 1)] = value
            elif key.startswith('neck.fpn_convs.4.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.4.conv.', 'le_fpn.P7_2.', 1)] = value
                mapped_weights[key.replace('neck.fpn_convs.4.conv.', 'ce_fpn.P7_2.', 1)] = value
            # if not using SimplifiedClassificationModel and SimplifiedRegressionModel map the following
            if not Hyperparameter.use_simple_reg_head:
                if key.startswith('bbox_head.reg_convs.0.conv.'):
                    mapped_weights[key.replace('bbox_head.reg_convs.0.conv.', 'regressionModel.conv1.', 1)] = value
                elif key.startswith('bbox_head.reg_convs.1.conv.'):
                    mapped_weights[key.replace('bbox_head.reg_convs.1.conv.', 'regressionModel.conv2.', 1)] = value
                elif key.startswith('bbox_head.reg_convs.2.conv.'):
                    mapped_weights[key.replace('bbox_head.reg_convs.2.conv.', 'regressionModel.conv3.', 1)] = value
                elif key.startswith('bbox_head.reg_convs.3.conv.'):
                    mapped_weights[key.replace('bbox_head.reg_convs.3.conv.', 'regressionModel.conv4.', 1)] = value
                elif key.startswith('bbox_head.retina_reg.'):
                    mapped_weights[key.replace('bbox_head.retina_reg.', 'regressionModel.output.', 1)] = value
            if not Hyperparameter.use_simple_cls_head:
                if key.startswith('bbox_head.cls_convs.0.conv.'):
                    mapped_weights[key.replace('bbox_head.cls_convs.0.conv.', 'classificationModel.conv1.', 1)] = value
                elif key.startswith('bbox_head.cls_convs.1.conv.'):
                    mapped_weights[key.replace('bbox_head.cls_convs.1.conv.', 'classificationModel.conv2.', 1)] = value
                elif key.startswith('bbox_head.cls_convs.2.conv.'):
                    mapped_weights[key.replace('bbox_head.cls_convs.2.conv.', 'classificationModel.conv3.', 1)] = value
                elif key.startswith('bbox_head.cls_convs.3.conv.'):
                    mapped_weights[key.replace('bbox_head.cls_convs.3.conv.', 'classificationModel.conv4.', 1)] = value
                elif key.startswith('bbox_head.retina_cls.'):
                    if Hyperparameter.num_classes == 2:
                        mapped_weights[key.replace('bbox_head.retina_cls.', 'classificationModel.output.', 1)] = value

        model_dict.update(mapped_weights)
        self.load_state_dict(model_dict)
        print("Pre-trained weights loaded successfully into dual streams and detection heads.")

    def __str__(self):
        """
        Provides a detailed string representation of the model's architecture,
        including layer names, parameter counts, shapes, and trainability.
        """
        # Create lists to hold information for formatting
        layer_info = []
        total_params = 0
        trainable_params = 0

        # Iterate through all named parameters to gather details
        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            
            layer_info.append({
                "name": name,
                "params": num_params,
                "shape": list(param.shape),
                "trainable": "Yes" if param.requires_grad else "No"
            })

        # Return early if there are no parameters
        if not layer_info:
            return "Model has no parameters."

        # Determine column widths for nice formatting
        max_name_len = max(len(info["name"]) for info in layer_info)
        max_params_len = max(len(f'{info["params"]:,}') for info in layer_info)
        max_shape_len = max(len(str(info["shape"])) for info in layer_info)

        # Build the header
        header = (
            f"{'Layer Name':<{max_name_len}} | "
            f"{'Parameters':>{max_params_len}} | "
            f"{'Shape':<{max_shape_len}} | "
            f"Trainable\n"
        )
        separator = (
            f"{'-' * max_name_len}-+-"
            f"{'-' * max_params_len}-+-"
            f"{'-' * max_shape_len}-+-"
            f"----------\n"
        )
        
        # Build the string with a table of layers
        s = header + separator
        for info in layer_info:
            s += (
                f"{info['name']:<{max_name_len}} | "
                f"{info['params']:,>{max_params_len}} | "
                f"{str(info['shape']):<{max_shape_len}} | "
                f"{info['trainable']}\n"
            )
        
        s += separator
        
        # Add summary statistics
        s += f"Total Parameters:     {total_params:,}\n"
        s += f"Trainable Parameters: {trainable_params:,}\n"
        if total_params > 0:
            percentage = 100 * trainable_params / total_params
            s += f"Trainable Percentage: {percentage:.2f}%\n"
        
        s += "=" * (len(separator.strip()) if separator else 100) + "\n"

        return s
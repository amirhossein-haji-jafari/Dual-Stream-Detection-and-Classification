import torch
import torch.nn as nn

# Re-using the same components from your project
from .retinanet import (Resnet, PyramidFeatures, RegressionModel, ClassificationModel, SimplifiedRegressionModel, SimplifiedClassificationModel)
from .retinanet_training import FocalLoss
from .utils.anchors import Anchors
from ..immutables import Hyperparameter

class SingleStreamRetinaNet(nn.Module):
    def __init__(self, num_classes, phi=2, pretrained_backbone=None):
        super().__init__()
        # phi=2 corresponds to ResNet-50
        fpn_sizes = {2: [512, 1024, 2048]}[phi]

        # 1. Single-Stream Backbone and FPN
        self.backbone = Resnet(phi, pretrained=pretrained_backbone)
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2],
                                   p3_2_dropout_rate=Hyperparameter.fpn_p3_2_dropout_rate,
                                   p4_2_dropout_rate=Hyperparameter.fpn_p4_2_dropout_rate,
                                   p5_2_dropout_rate=Hyperparameter.fpn_p5_2_dropout_rate,
                                   p6_dropout_rate=Hyperparameter.fpn_p6_dropout_rate,
                                   p7_dropout_rate=Hyperparameter.fpn_p7_dropout_rate)

        # 2. Detection Head
        # Note: We keep the simplified/standard head logic for a fair comparison
        if Hyperparameter.use_simple_cls_head:
            self.classificationModel = SimplifiedClassificationModel(256, Hyperparameter.simple_cls_dropout_rate, num_classes=num_classes)
        else:
            self.classificationModel = ClassificationModel(256, Hyperparameter.cls_dropout_rate, num_classes=num_classes)

        if Hyperparameter.use_simple_reg_head:
            self.regressionModel = SimplifiedRegressionModel(256, Hyperparameter.simple_reg_dropout_rate)
        else:
            self.regressionModel = RegressionModel(256, Hyperparameter.reg_dropout_rate)

        # 3. Anchors and Loss functions
        self.anchors = Anchors()
        self.focal_loss = FocalLoss()

    def forward(self, image, annotations=None, cuda=True):
        # --- Backbone and FPN pass ---
        c3, c4, c5 = self.backbone(image)
        features = self.fpn([c3, c4, c5])
        
        # --- Detection Head ---
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(features)
        
        # --- Loss Calculation (Training/Validation) ---
        if annotations is not None:
            _, cls_loss, reg_loss = self.focal_loss(classification, regression, anchors, annotations, cuda=cuda, alpha=Hyperparameter.focal_loss_alpha, gamma=Hyperparameter.focal_loss_gamma)
            return cls_loss, reg_loss
        
        # --- Inference ---
        else:
            return regression, classification, anchors

    def load_pretrained_weights(self, pth_path):
        print(f"Loading pre-trained weights from {pth_path} for SingleStreamRetinaNet")
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
                mapped_weights[key.replace('backbone.', 'backbone.model.', 1)] = value
            elif key.startswith('neck.lateral_convs.0.conv.'):
                mapped_weights[key.replace('neck.lateral_convs.0.conv.', 'fpn.P3_1.', 1)] = value
            elif key.startswith('neck.lateral_convs.1.conv.'):
                mapped_weights[key.replace('neck.lateral_convs.1.conv.', 'fpn.P4_1.', 1)] = value
            elif key.startswith('neck.lateral_convs.2.conv.'):
                mapped_weights[key.replace('neck.lateral_convs.2.conv.', 'fpn.P5_1.', 1)] = value
            elif key.startswith('neck.fpn_convs.0.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.0.conv.', 'fpn.P3_2.', 1)] = value
            elif key.startswith('neck.fpn_convs.1.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.1.conv.', 'fpn.P4_2.', 1)] = value
            elif key.startswith('neck.fpn_convs.2.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.2.conv.', 'fpn.P5_2.', 1)] = value
            elif key.startswith('neck.fpn_convs.3.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.3.conv.', 'fpn.P6.', 1)] = value
            elif key.startswith('neck.fpn_convs.4.conv.'):
                mapped_weights[key.replace('neck.fpn_convs.4.conv.', 'fpn.P7_2.', 1)] = value
            
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
        print("Pre-trained weights loaded successfully into single stream model.")
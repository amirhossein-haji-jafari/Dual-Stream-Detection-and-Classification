import torch
import torch.nn as nn
from transformers import AutoModel

from ..immutables import Hyperparameter
from ..fusion import ConditionalFusion, SynergisticFusion, ReluAddFusion, ConvReluFusion
from .resnet import Resnet
class ImageClassifierHead(nn.Module):
    """ Image-level classification head using fused C5 features. """
    def __init__(self, in_channels, num_classes, dropout_rate):
        super().__init__()

        print(f"Using '{Hyperparameter.fusion_method_cls}' feature fusion method.")
        match Hyperparameter.fusion_method_cls:
            case 'conditional':
                self.fusion = ConditionalFusion(in_channels)
            case 'synergistic':
                self.fusion = SynergisticFusion(in_channels)
            case 'relu_add':
                self.fusion = ReluAddFusion(in_channels)
            case 'conv-relu':
                self.fusion = ConvReluFusion(in_channels)
            case _:
                raise ValueError(
                    f"Fusion method is invalid. Expected 'conditional', 'synergistic', 'relu_add', or 'conv-relu', "
                    f"but got '{Hyperparameter.fusion_method_cls}'."
                )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)

        if num_classes == 2:
            self.fc = nn.Linear(in_channels, 1) # 1 output for Malignant/Benign
        elif num_classes == 3:
            self.fc = nn.Linear(in_channels, 3) # 3 outputs for Malignant/Benign/Normal
        else:
            raise ValueError(f"num_classes must be 2 or 3 for image classification. Got {num_classes}.")

    def forward(self, c5_le, c5_ce):
        c5_le = self.pool(c5_le)
        c5_ce = self.pool(c5_ce)
        x = self.fusion(c5_le, c5_ce)
        # x = self.pool(x)
        x = torch.flatten(x, 1)
        # Apply dropout before the final fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DualStreamClassification(nn.Module):
    def __init__(self, num_classes, backbone = "ianpan_mammoscreen"):
        super().__init__()
        # 1. Dual-Stream Backbone
        if backbone == 'resnet-of-retinanet':
            print("Using ResNet-50 backbone from RetinaNet.")
            self.le_backbone = Resnet()
            self.ce_backbone = Resnet()
            # 2. Image Classification Head
            self.imageClassifier = ImageClassifierHead(2048, num_classes, Hyperparameter.img_cls_dropout_rate)
        elif backbone == 'ianpan_mammoscreen':
            print("Using ianpan_mammoscreen backbone.")
            model_le = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True).net2
            model_ce = AutoModel.from_pretrained("ianpan/mammoscreen", trust_remote_code=True).net2
            self.le_backbone = nn.Sequential(*list(model_le.children())[:-1])
            self.ce_backbone = nn.Sequential(*list(model_ce.children())[:-1])
            # 2. Image Classification Head
            self.imageClassifier = ImageClassifierHead(1280, num_classes, Hyperparameter.img_cls_dropout_rate)
            del model_le, model_ce

        self.num_classes = num_classes

        # 3. Loss function
        if num_classes == 2:
            self.img_cls_loss_fn = nn.BCEWithLogitsLoss()
        elif num_classes == 3:
            self.img_cls_loss_fn = nn.CrossEntropyLoss()

    def forward(self, le_image, ce_image, image_class_target=None, is_validating=False):
        # --- Backbone pass ---
        # c5_le = self.le_backbone(le_image[:, 0:1, :, :])
        # c5_ce = self.ce_backbone(ce_image[:, 0:1, :, :])
        le = self.le_backbone(le_image)
        ce = self.ce_backbone(ce_image)

        image_class_pred = self.imageClassifier(le, ce)

        # --- Loss Calculation (Training/Validation) ---
        if (image_class_target is not None) and (is_validating == False):
            # in Training mode
            img_loss = self.img_cls_loss_fn(image_class_pred.squeeze(-1), image_class_target.float())
            return img_loss
        elif (image_class_target is not None) and (is_validating == True):
            # in Validation mode
            img_loss = self.img_cls_loss_fn(image_class_pred.squeeze(-1), image_class_target.float())
            if self.num_classes == 2:
                preds_prob = torch.sigmoid(image_class_pred)
            else: # num_classes == 3
                preds_prob = torch.softmax(image_class_pred, dim=1)
            
            return img_loss, preds_prob
        elif image_class_target is None and (is_validating == False):
            # in Testing mode
            if self.num_classes == 3:
                return torch.softmax(image_class_pred, dim=1)
            else: # num_classes == 2
                return torch.sigmoid(image_class_pred)
        else:
            raise ValueError("image_class_target must be provided during training and validation.")


    def load_pretrained_weights(self, weights_path):
        print(f"Loading pre-trained weights of RetinaNet's ResNet-50 from {weights_path}")
        device = next(self.parameters()).device
        model_dict = self.state_dict()
        mapped_weights = {}

        pretrained_dict = torch.load(weights_path, map_location=device)["state_dict"]
        for key, value in pretrained_dict.items():
            if key.startswith('backbone.'):
                mapped_weights[key.replace('backbone.', 'le_backbone.model.', 1)] = value
                mapped_weights[key.replace('backbone.', 'ce_backbone.model.', 1)] = value
     
        model_dict.update(mapped_weights)
        self.load_state_dict(model_dict)
        print("Pre-trained weights loaded successfully into dual streams.")

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
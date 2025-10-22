import os
import unittest
from unittest.mock import patch

import torch

# --- Imports from your project files ---
from ..dual_stream_retinanet import DualStreamRetinaNet
from ...immutables import ProjectPaths


class TestDualStreamWeightLoading(unittest.TestCase):
    """
    Test suite to verify the correct loading of pretrained weights from a standard
    RetinaNet model into the DualStreamRetinaNet architecture.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load the source pretrained weights dictionary once for all tests in this class.
        This is efficient as it avoids reading the file from disk for every test.
        """
        cls.pretrained_path = ProjectPaths.pretrained_retinanet
        if not os.path.exists(cls.pretrained_path):
            raise FileNotFoundError(
                f"Pretrained model file not found at {cls.pretrained_path}. "
                "Please ensure the path is correct in immutables.py and the file exists."
            )
        # Load the state dictionary from the .pth file
        cls.source_state_dict = torch.load(cls.pretrained_path, map_location="cpu")["state_dict"]

    def test_backbone_weights_are_loaded_correctly(self):
        """
        Verifies that both the Low-Energy (LE) and Contrast-Enhanced (CE) backbones
        are correctly and identically loaded from the source backbone weights.
        """
        model = DualStreamRetinaNet(num_classes=1, phi=2)
        model.load_pretrained_weights(self.pretrained_path)
        model_state_dict = model.state_dict()

        # Check all keys from the source backbone
        for source_key, source_tensor in self.source_state_dict.items():
            if source_key.startswith("backbone."):
                # Construct the corresponding destination keys for both streams
                dest_key_le = source_key.replace("backbone.", "le_backbone.model.", 1)
                dest_key_ce = source_key.replace("backbone.", "ce_backbone.model.", 1)

                with self.subTest(source_key=source_key):
                    # Verify LE backbone tensor matches the source
                    self.assertIn(dest_key_le, model_state_dict)
                    dest_tensor_le = model_state_dict[dest_key_le]
                    torch.testing.assert_close(source_tensor, dest_tensor_le,
                                               msg=f"Weight mismatch for {dest_key_le}")

                    # Verify CE backbone tensor matches the source
                    self.assertIn(dest_key_ce, model_state_dict)
                    dest_tensor_ce = model_state_dict[dest_key_ce]
                    torch.testing.assert_close(source_tensor, dest_tensor_ce,
                                               msg=f"Weight mismatch for {dest_key_ce}")

    def test_fpn_weights_are_loaded_correctly(self):
        """
        Verifies that both the LE and CE Feature Pyramid Networks (FPNs)
        are correctly and identically loaded from the source FPN (neck) weights.
        """
        model = DualStreamRetinaNet(num_classes=1, phi=2)
        model.load_pretrained_weights(self.pretrained_path)
        model_state_dict = model.state_dict()
        
        # This dictionary maps source prefixes to their destination counterparts
        fpn_mappings = {
            'neck.lateral_convs.0.conv.': 'P3_1.', 'neck.lateral_convs.1.conv.': 'P4_1.',
            'neck.lateral_convs.2.conv.': 'P5_1.', 'neck.fpn_convs.0.conv.': 'P3_2.',
            'neck.fpn_convs.1.conv.': 'P4_2.', 'neck.fpn_convs.2.conv.': 'P5_2.',
            'neck.fpn_convs.3.conv.': 'P6.', 'neck.fpn_convs.4.conv.': 'P7_2.',
        }

        for source_key, source_tensor in self.source_state_dict.items():
            for source_prefix, dest_prefix in fpn_mappings.items():
                if source_key.startswith(source_prefix):
                    suffix = source_key[len(source_prefix):]
                    dest_key_le = f"le_fpn.{dest_prefix}{suffix}"
                    dest_key_ce = f"ce_fpn.{dest_prefix}{suffix}"
                    
                    with self.subTest(source_key=source_key):
                        self.assertIn(dest_key_le, model_state_dict)
                        torch.testing.assert_close(source_tensor, model_state_dict[dest_key_le])
                        
                        self.assertIn(dest_key_ce, model_state_dict)
                        torch.testing.assert_close(source_tensor, model_state_dict[dest_key_ce])
                    break

    @patch('dual_stream_retinanet.Hyperparameter.use_simple_reg_head', False)
    @patch('dual_stream_retinanet.Hyperparameter.use_simple_cls_head', False)
    @patch('dual_stream_retinanet.Hyperparameter.num_classes', 2) # Pretrained model's final layer has 2 classes
    def test_standard_detection_heads_are_loaded(self):
        """
        Verifies that the standard (non-simple) detection heads are loaded correctly
        when the corresponding Hyperparameter flags are set to False.
        """
        model = DualStreamRetinaNet(num_classes=2, phi=2)
        model.load_pretrained_weights(self.pretrained_path)
        model_state_dict = model.state_dict()

        # Explicit mapping reflecting the logic in load_pretrained_weights
        head_mappings = {
            'bbox_head.reg_convs.0.conv.': 'regressionModel.conv1.',
            'bbox_head.reg_convs.1.conv.': 'regressionModel.conv2.',
            'bbox_head.reg_convs.2.conv.': 'regressionModel.conv3.',
            'bbox_head.reg_convs.3.conv.': 'regressionModel.conv4.',
            'bbox_head.retina_reg.': 'regressionModel.output.',
            'bbox_head.cls_convs.0.conv.': 'classificationModel.conv1.',
            'bbox_head.cls_convs.1.conv.': 'classificationModel.conv2.',
            'bbox_head.cls_convs.2.conv.': 'classificationModel.conv3.',
            'bbox_head.cls_convs.3.conv.': 'classificationModel.conv4.',
            'bbox_head.retina_cls.': 'classificationModel.output.',
        }

        for source_key, source_tensor in self.source_state_dict.items():
            for source_prefix, dest_prefix in head_mappings.items():
                if source_key.startswith(source_prefix):
                    suffix = source_key[len(source_prefix):]
                    dest_key = f"{dest_prefix}{suffix}"
                    
                    with self.subTest(source_key=source_key):
                        self.assertIn(dest_key, model_state_dict, f"Key '{dest_key}' not found in model.")
                        dest_tensor = model_state_dict[dest_key]
                        torch.testing.assert_close(source_tensor, dest_tensor)
                    break

    @patch('dual_stream_retinanet.Hyperparameter.fusion_method', 'conv-relu')
    def test_fusion_layers_are_not_loaded_from_source(self):
        """
        Verifies that custom layers (fusion modules) are not affected by weight loading.
        We patch the fusion_method to 'conv-relu' to ensure we test a module
        with trainable convolution weights.
        """
        # Create a fresh, randomly initialized model for comparison
        fresh_model = DualStreamRetinaNet(num_classes=1, phi=2)
        
        # Create a model and load weights into it
        loaded_model = DualStreamRetinaNet(num_classes=1, phi=2)
        loaded_model.load_pretrained_weights(self.pretrained_path)

        for i in range(5): # Iterate through the 5 fusion modules
            fresh_weight = fresh_model.fusion_modules[i].conv.weight
            loaded_weight = loaded_model.fusion_modules[i].conv.weight
            
            # The weights should be different, since the loaded model's fusion layer
            # should have been initialized differently from the fresh one and not overwritten.
            self.assertFalse(torch.equal(fresh_weight, loaded_weight),
                             f"Fusion module {i} conv weights were unexpectedly identical to a fresh model.")
            
            # More robustly, ensure the fusion layer weights are not equal to ANY source weight
            is_equal_to_any_source = any(torch.equal(loaded_weight, source_tensor) 
                                         for source_tensor in self.source_state_dict.values())
            self.assertFalse(is_equal_to_any_source,
                            f"Fusion module {i} weights were incorrectly loaded from a source layer.")


if __name__ == "__main__":
    unittest.main()
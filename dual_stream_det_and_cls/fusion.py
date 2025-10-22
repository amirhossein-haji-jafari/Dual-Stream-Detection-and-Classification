import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    """ Fuses features by learning a gate to weigh the two streams. """
    def __init__(self, in_channels, dropout_rate=0.2):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        
        # This layer learns the gate
        self.gate_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
        # This layer processes the final fused features
        # self.feature_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        # Concatenate features to compute the gate
        combined = torch.cat([x1_norm, x2_norm], dim=1)
        
        # Compute the gate and apply sigmoid to get weights between 0 and 1
        gate = torch.sigmoid(self.gate_conv(combined))

        # Apply the gate to fuse the features
        # Learns a soft switch: take 'gate' amount from x1 and '1-gate' from x2
        x_fused_gated = (x1_norm * gate) + (x2_norm * (1 - gate))
        x_out = self.dropout(x_fused_gated)

        # Further process the fused features
        # x_out = self.feature_conv(x_fused_gated)
        # x_out = self.relu(x_out)
        # x_out = self.dropout(x_out)
        
        return x_out

class MaxFusion(nn.Module):
    """ Fuses feature maps by taking the element-wise maximum. """
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        
        # Take the strongest activation at each point
        x_fused = torch.maximum(x1_norm, x2_norm)
        
        return x_fused
class ConvReluFusion(nn.Module):
    """ Fuses feature maps from two streams using LayerNorm, Concat, and 1x1 Conv. """
    def __init__(self, in_channels, dropout_rate = 0.0):
        super().__init__()
        # LayerNorm on channel dimension. nn.GroupNorm(1, C) is equivalent.
        self.norm = nn.GroupNorm(1, in_channels) 
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        x_cat = torch.cat([x1_norm, x2_norm], dim=1)
        x_fused = self.conv(x_cat)
        x_fused = self.relu(x_fused)
        x_fused = self.dropout(x_fused)
        return x_fused

class ReluAddFusion(nn.Module):
    """ Fuses feature maps from two streams using LayerNorm and element-wise addition. """
    def __init__(self, in_channels):
        super().__init__()
        # LayerNorm on channel dimension. nn.GroupNorm(1, C) is equivalent.
        # This helps stabilize the features from each stream before adding them.
        self.norm = nn.GroupNorm(1, in_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        # Normalize each stream independently
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)
        
        # Apply ReLU to each stream to remove negative values
        x1_relu = self.relu(x1_norm)
        x2_relu = self.relu(x2_norm)
        # Fuse by simple element-wise addition
        x_fused = x1_relu + x2_relu
        # x_fused = x1_norm + x2_norm
        
        return x_fused

class ConditionalFusion(nn.Module):
    """
    Fuses features based on a conditional strategy:
    - If signs of features from both streams agree, they are added.
    - If signs disagree, the maximum of the two is taken.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        # Condition: Do the signs of the features agree?
        # (positive * positive >= 0) and (negative * negative >= 0)
        # (positive * negative < 0)
        signs_agree_mask = (x1_norm * x2_norm) >= 0

        # Action for agreement: Add the features
        fused_when_agree = x1_norm + x2_norm

        # Action for disagreement: Take the maximum (which will always be the positive value)
        fused_when_disagree = torch.maximum(x1_norm, x2_norm)

        # Apply the condition to select the appropriate action for each element
        x_fused = torch.where(signs_agree_mask, fused_when_agree, fused_when_disagree)
        
        return x_fused

class SynergisticFusion(nn.Module):
    """
    Fuses features with a strong bias towards positive evidence to improve recall.
    - If both features are positive, they are added to create a synergistic signal.
    - In all other cases (disagreement or both negative), the maximum value is taken.
      This preserves positive signals and weakens shared negative signals.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)

    def forward(self, x1, x2):
        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        # Condition: Are both features strictly positive?
        # We use the element-wise logical AND operator '&'
        both_positive_mask = (x1_norm > 0) & (x2_norm > 0)

        # Action 1 (Synergy): Add the features
        fused_when_synergy = x1_norm + x2_norm

        # Action 2 (All other cases): Take the maximum
        fused_otherwise = torch.maximum(x1_norm, x2_norm)

        # Apply the condition to select the appropriate action for each element
        x_fused = torch.where(both_positive_mask, fused_when_synergy, fused_otherwise)
        
        return x_fused
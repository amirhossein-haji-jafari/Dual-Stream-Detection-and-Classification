import torch.nn as nn
from torchvision.models import resnet50
class Resnet(nn.Module):
    """
    ResNet-50 Backbone of RetinaNet.
    """
    def __init__(self, pretrained=None):
        super(Resnet, self).__init__()
        model = resnet50(weights = pretrained)
        del model.avgpool
        del model.fc
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        feat1 = self.model.layer2(x)
        feat2 = self.model.layer3(feat1)
        feat3 = self.model.layer4(feat2)

        return  feat3 # C5 features
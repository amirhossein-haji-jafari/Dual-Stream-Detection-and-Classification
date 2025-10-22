import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.anchors import Anchors

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, p3_2_dropout_rate, p4_2_dropout_rate, p5_2_dropout_rate, p6_dropout_rate, p7_dropout_rate, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_1_relu = nn.ReLU() # Enhanced RetinaNet from https://doi.org/10.2147/JMDH.S493873
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # dropout layers to be reused
        self.P3_2_dropout = nn.Dropout2d(p=p3_2_dropout_rate)
        self.P4_2_dropout = nn.Dropout2d(p=p4_2_dropout_rate)
        self.P5_2_dropout = nn.Dropout2d(p=p5_2_dropout_rate)
        self.P6_dropout = nn.Dropout2d(p=p6_dropout_rate)
        self.P7_dropout = nn.Dropout2d(p=p7_dropout_rate)
    def forward(self, inputs):
        C3, C4, C5 = inputs
        _, _, h4, w4 = C4.size()
        _, _, h3, w3 = C3.size()

        # 75,75,512 -> 75,75,256
        P3_x = self.P3_1(C3)
        # 38,38,1024 -> 38,38,256
        P4_x = self.P4_1(C4)
        # 19,19,2048 -> 19,19,256
        P5_x = self.P5_1(C5)

        # 19,19,256 -> 38,38,256
        P5_upsampled_x = F.interpolate(P5_x, size=(h4, w4))
        # 38,38,256 + 38,38,256 -> 38,38,256
        P4_x = P5_upsampled_x + P4_x
        # 38,38,256 -> 75,75,256
        P4_upsampled_x = F.interpolate(P4_x, size=(h3, w3))
        # 75,75,256 + 75,75,256 -> 75,75,256
        P3_x = P3_x + P4_upsampled_x

        # 75,75,256 -> 75,75,256
        P3_x = self.P3_2(P3_x)
        P3_x = self.P3_2_dropout(P3_x)
        # 38,38,256 -> 38,38,256
        P4_x = self.P4_2(P4_x)
        P4_x = self.P4_2_dropout(P4_x)
        # 19,19,256 -> 19,19,256
        P5_x = self.P5_1_relu(P5_x) # Enhanced RetinaNet from https://doi.org/10.2147/JMDH.S493873
        P5_x = self.P5_2(P5_x)
        P5_x = self.P5_2_dropout(P5_x)

        # 19,19,2048 -> 10,10,256
        P6_x = self.P6(C5)
        P6_x = self.P6_dropout(P6_x)

        P7_x = self.P7_1(P6_x)
        # 10,10,256 -> 5,5,256
        P7_x = self.P7_2(P7_x)
        P7_x = self.P7_dropout(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, dropout_rate, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        # Dropout2d
        self.drop2  = nn.Dropout2d(p=dropout_rate)

        self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3   = nn.ReLU()

        self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4   = nn.ReLU()

        # Dropout2d
        self.drop4  = nn.Dropout2d(p=dropout_rate) 

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.drop2(out) # Apply dropout

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.drop4(out) # Apply dropout

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class SimplifiedRegressionModel(nn.Module):
    def __init__(self, num_features_in, dropout_rate, num_anchors=9, feature_size=256):
        super(SimplifiedRegressionModel, self).__init__()

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        # Add Dropout
        self.drop1 = nn.Dropout2d(p=dropout_rate)

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.drop1(out) # Apply dropout

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)
class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, dropout_rate, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        # Add Dropout2d
        self.drop2  = nn.Dropout2d(p=dropout_rate) 

        self.conv3  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3   = nn.ReLU()

        self.conv4  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4   = nn.ReLU()

        # Add Dropout2d
        self.drop4  = nn.Dropout2d(p=dropout_rate)

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.drop2(out) # Apply dropout

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.drop4(out) # Apply dropout

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, height, width, channels = out1.shape

        out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class SimplifiedClassificationModel(nn.Module):
    def __init__(self, num_features_in, dropout_rate, num_anchors=9, num_classes=80, feature_size=128):
        super(SimplifiedClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1  = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1   = nn.ReLU()
        
        # Add Dropout
        self.drop1 = nn.Dropout2d(p=dropout_rate) 

        self.conv2  = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2   = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.drop1(out) # Apply dropout

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, height, width, channels = out1.shape

        out2 = out1.view(batch_size, height, width, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class Resnet(nn.Module):
    def __init__(self, phi, pretrained=None):
        super(Resnet, self).__init__()
        self.edition = [resnet18, resnet34, resnet50, resnet101, resnet152]
        model = self.edition[phi](weights = pretrained)
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

        return [feat1, feat2, feat3]

class retinanet(nn.Module):
    def __init__(self, num_classes, phi, pretrained=None):
        super(retinanet, self).__init__()
        self.pretrained = pretrained

        #  C3、C4、C5
        self.backbone_net = Resnet(phi, pretrained)
        fpn_sizes = {
            0: [128, 256, 512],
            1: [128, 256, 512],
            2: [512, 1024, 2048],
            3: [512, 1024, 2048],
            4: [512, 1024, 2048],
        }[phi]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel        = RegressionModel(256)
        self.classificationModel    = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()
        self._init_weights()

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

    def forward(self, inputs):
        p3, p4, p5 = self.backbone_net(inputs)
        # features: p3,p4,p5,p6,p7
        features = self.fpn([p3, p4, p5])

        #  prediction
        regression      = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification  = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(features)

        return features, regression, classification, anchors

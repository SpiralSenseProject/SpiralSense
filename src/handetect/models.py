#######################################################
# This file stores all the models used in the project.#
#######################################################

# Import all models from torchvision.models
from torchvision.models import (
    squeezenet1_0,
    SqueezeNet1_0_Weights,
    squeezenet1_1,
    SqueezeNet1_1_Weights,
    shufflenet_v2_x2_0,
    ShuffleNet_V2_X2_0_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b1,
    EfficientNet_B1_Weights,
    efficientnet_b2,
    EfficientNet_B2_Weights,
    efficientnet_b3,
    EfficientNet_B3_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    googlenet,
    GoogLeNet_Weights,
    mobilenet_v2,
    MobileNet_V2_Weights,
)
import torch.nn as nn
import torch
import torch.nn.functional as F

class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),  # Sigmoid activation to produce attention scores
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SqueezeNet1_0WithSE(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(SqueezeNet1_0WithSE, self).__init__()
        squeezenet = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
        self.features = squeezenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Add dropout layer with the specified probability

        # Adjust channel for SqueezeNet1.0 (original SqueezeNet1.0 has 1000 classes)
        num_classes_squeezenet1_0 = 7

        # Add Squeeze-and-Excitation block
        self.se_block = SE_Block(
            channel=num_classes_squeezenet1_0
        )  # Adjust channel as needed

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # x = self.se_block(x)  # Apply the SE block
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = torch.flatten(x, 1)
        return x


class SqueezeNet1_1WithSE(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.2):
        super(SqueezeNet1_1WithSE, self).__init__()
        squeezenet = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        self.features = squeezenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Add dropout layer with the specified probability

        # Add Squeeze-and-Excitation block
        self.se_block = SE_Block(channel=num_classes)  # Adjust channel as needed

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.se_block(x)  # Apply the SE block
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = torch.flatten(x, 1)
        return x


class EfficientNetB2WithDropout(nn.Module):
    #  0.00022015769999619205
    def __init__(self, num_classes, dropout_prob=0.2):
        super(EfficientNetB2WithDropout, self).__init__()
        efficientnet = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        self.features = efficientnet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(1408, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Add dropout layer with the specified probability

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = torch.flatten(x, 1)
        return x


class EfficientNetB3WithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.2):
        super(EfficientNetB3WithDropout, self).__init__()
        efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = efficientnet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(1536, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Add dropout layer with the specified probability

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = torch.flatten(x, 1)
        return x


class MobileNetV2WithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.2):
        super(MobileNetV2WithDropout, self).__init__()
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(1280, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Add dropout layer with the specified probability

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = torch.flatten(x, 1)
        return x


class GoogLeNetWithSE(nn.Module):
    def __init__(self, num_classes):
        super(GoogLeNetWithSE, self).__init__()
        googlenet = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        # self.features = googlenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Add Squeeze-and-Excitation block
        self.se_block = SE_Block(channel=num_classes)  # Adjust channel as needed

    def forward(self, x):
        # x = self.features(x)
        x = self.classifier(x)
        x = self.se_block(x)  # Apply the SE block
        x = torch.flatten(x, 1)
        return x

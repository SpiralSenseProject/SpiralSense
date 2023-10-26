from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import torch
import torch.nn as nn


class EfficientNetB3WithNorm(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3WithNorm, self).__init__()
        efficientnet = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = efficientnet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(1536, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x

import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from models import *
import torch.nn as nn
from torchvision.models import (
    squeezenet1_0,
    SqueezeNet1_0_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
from torchvision.models import squeezenet1_0

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 5.488903014780378e-05
STEP_SIZE = 10
GAMMA = 0.3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PRINT = 100
TASK = 1
RAW_DATA_DIR = r"data/train/raw/Task "
AUG_DATA_DIR = r"data/train/augmented/Task "
EXTERNAL_DATA_DIR = r"data/train/external/Task "
TEMP_DATA_DIR = "data/temp/"
NUM_CLASSES = 7
EARLY_STOPPING_PATIENCE = 20
CLASSES = [
    "Alzheimer Disease",
    "Cerebral Palsy",
    "Dystonia",
    "Essential Tremor",
    "Healthy",
    "Huntington Disease",
    "Parkinson Disease",
]
MODEL_SAVE_PATH = r"output/checkpoints/model.pth"


class SqueezeNet1_0WithDropout(nn.Module):
    def __init__(self, num_classes=1000):
        super(SqueezeNet1_0WithDropout, self).__init__()
        squeezenet = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
        self.features = squeezenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x


# class ShuffleNetV2WithDropout(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(ShuffleNetV2WithDropout, self).__init__()
#         shufflenet = shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights)
#         self.features = shufflenet.features
#         self.classifier = nn.Sequential(
#             nn.Conv2d(1024, num_classes, kernel_size=1),
#             nn.BatchNorm2d(num_classes),  # add batch normalization
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         x = torch.flatten(x, 1)
#         return x


class MobileNetV3SmallWithDropout(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3SmallWithDropout, self).__init__()
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights)
        self.features = mobilenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(576, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),  # add batch normalization
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x


class ResNet18WithNorm(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18WithNorm, self).__init__()
        resnet = resnet18(pretrained=False)
        self.features = nn.Sequential(
            *list(resnet.children())[:-2]
        )  # Remove last 2 layers (avgpool and fc)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
            nn.BatchNorm2d(num_classes),  # Add batch normalization
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x


MODEL = SqueezeNet1_0WithDropout(num_classes=7)
print(CLASSES)

preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),  # Convert to tensor
        # Normalize 3 channels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label

import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from models import *
import torch.nn as nn
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
)

import torch.nn.functional as F
from pytorchcv.model_provider import get_model as ptcv_get_model

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1.4257700984917018e-05
STEP_SIZE = 10
GAMMA = 0.6
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PRINT = 100
TASK = 1
RAW_DATA_DIR = r"data/train/raw/Task "
AUG_DATA_DIR = r"data/train/augmented/Task "
EXTERNAL_DATA_DIR = r"data/train/external/Task "
COMBINED_DATA_DIR = r"data/train/combined/Task "
TEMP_DATA_DIR = "data/temp/"
NUM_CLASSES = 7
LABEL_SMOOTHING_EPSILON = 0.1
MIXUP_ALPHA = 0.2
EARLY_STOPPING_PATIENCE = 10
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
    def __init__(self, num_classes, dropout_prob=0.5):
        super(SqueezeNet1_0WithDropout, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = torch.flatten(x, 1)
        return x


class SqueezeNet1_1WithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(SqueezeNet1_1WithDropout, self).__init__()
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = torch.flatten(x, 1)
        return x


class ShuffleNetV2WithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(ShuffleNetV2WithDropout, self).__init__()
        shufflenet = shufflenet_v2_x2_0(weights=ShuffleNet_V2_X2_0_Weights.DEFAULT)
        self.features = shufflenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, kernel_size=1),
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


class MobileNetV3SmallWithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(MobileNetV3SmallWithDropout, self).__init__()
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = mobilenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(576, num_classes, kernel_size=1),
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


class EfficientNetV2SmallWithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(EfficientNetV2SmallWithDropout, self).__init__()
        efficientnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.features = efficientnet.features
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

MODEL = EfficientNetV2SmallWithDropout(num_classes=7, dropout_prob=0.5)
# MODEL = ptcv_get_model("sqnxt23v5_w2", pretrained=False, num_classes=7)
print(CLASSES)

preprocess = transforms.Compose(
    [
        transforms.Resize((112, 112)),  # Resize to 112x112
        transforms.ToTensor(),  # Convert to tensor
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
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

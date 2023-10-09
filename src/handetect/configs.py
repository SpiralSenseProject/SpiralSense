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
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b1,
    EfficientNet_B1_Weights,
    efficientnet_b2,
    EfficientNet_B2_Weights,
)

import torch.nn.functional as F

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 5.8196208148896214e-05
STEP_SIZE = 10
GAMMA = 0.6
CUTMIX_ALPHA = 0.3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PRINT = 100
TASK = 1
WARMUP_EPOCHS = 5
RAW_DATA_DIR = r"data/train/raw/Task "
AUG_DATA_DIR = r"data/train/augmented/Task "
EXTERNAL_DATA_DIR = r"data/train/external/Task "
COMBINED_DATA_DIR = r"data/train/combined/Task "
TEMP_DATA_DIR = "data/temp/"
NUM_CLASSES = 7
LABEL_SMOOTHING_EPSILON = 0.1
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


class SE_Block(nn.Module):
    def __init__(self, channel, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
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
    def __init__(self, num_classes, dropout_prob=0.5):
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
    def __init__(self, num_classes, dropout_prob=0.5):
        super(EfficientNetB2WithDropout, self).__init__()
        efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
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


class ResNet18WithNorm(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18WithNorm, self).__init__()
        resnet = resnet18(pretrained=False)

        # Remove the last block (Block 4)
        self.features = nn.Sequential(
            *list(resnet.children())[:-1]  # Exclude the last block
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(
                512, num_classes
            ),  # Adjust input size for the fully connected layer
            nn.BatchNorm1d(num_classes),  # Add batch normalization
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x


MODEL = EfficientNetB2WithDropout(num_classes=NUM_CLASSES, dropout_prob=0.5)
MODEL_SAVE_PATH = r"output/checkpoints/" + MODEL.__class__.__name__ + ".pth"

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

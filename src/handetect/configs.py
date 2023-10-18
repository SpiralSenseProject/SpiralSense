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
    efficientnet_b3,
    EfficientNet_B3_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    googlenet,
    GoogLeNet_Weights,
    MobileNet_V2_Weights,
    mobilenet_v2,
)

import torch.nn.functional as F

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 150
WARMUP_EPOCHS = 5
LEARNING_RATE = 1.098582599143508e-04
STEP_SIZE = 10
GAMMA = 0.3
CUTMIX_ALPHA = 0.3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
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


class MobileNetV3LargeWithDropout(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.2):
        super(MobileNetV3LargeWithDropout, self).__init__()
        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.features = mobilenet.features
        self.classifier = nn.Sequential(
            nn.Conv2d(960, num_classes, kernel_size=1),
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


MODEL = EfficientNetB3WithDropout(num_classes=NUM_CLASSES)
MODEL_SAVE_PATH = r"output/checkpoints/" + MODEL.__class__.__name__ + ".pth"
# MODEL_SAVE_PATH = r"C:\Users\User\Downloads\bestsqueezenetSE.pth"
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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


def ensemble_predictions(models, image):
    all_predictions = []

    with torch.no_grad():
        for model in models:
            output = model(image)
            all_predictions.append(output)

    return torch.stack(all_predictions, dim=0).mean(dim=0)

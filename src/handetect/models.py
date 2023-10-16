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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array

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


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)  # Apply dropout during training
        x = self.fc3(x)
        return x


class EnsembleModel(nn.Module):
    def __init__(self, models, meta_learner):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.meta_learner = meta_learner

    def forward(self, x):
        # Extract features from individual models
        features = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                model_features = model(x)
            features.append(model_features)

        # Stack the features from individual models
        stacked_features = torch.cat(features, dim=1)

        # Pass the stacked features through the meta-learner
        ensemble_output = self.meta_learner(stacked_features)

        return ensemble_output



# Define the custom classifier class
class SqueezeNet1_0WithSEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, num_classes, dropout_prob=0.2):
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.model = None
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        # Check and validate input data
        X, y = check_X_y(X, y, multi_output=False)
        
        # Create and train the SqueezeNet1_0WithSE model
        self.model = SqueezeNet1_0WithSE(self.num_classes, self.dropout_prob)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # Convert labels to numeric values
        y = self.label_encoder.fit_transform(y)
        
        # Training loop
        for epoch in range(10):
            for inputs, labels in zip(X, y):
                inputs = torch.tensor(inputs).unsqueeze(0)
                labels = torch.tensor(labels)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        # Check and validate input data
        X = check_array(X)
        
        # Make predictions using the trained model
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for inputs in X:
                inputs = torch.tensor(inputs).unsqueeze(0)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.item())
        
        # Convert numeric predictions to original class labels
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions
    
# Define the ensemble model using a list of models
class WeightedVoteEnsemble(nn.Module):
    def __init__(self, models, weights):
        super(WeightedVoteEnsemble, self).__init__()
        self.models = models
        self.weights = weights

    def forward(self, x):
        predictions = [model(x) for model in self.models]
        weighted_predictions = torch.stack(
            [w * pred for w, pred in zip(self.weights, predictions)], dim=0
        )
        avg_predictions = weighted_predictions.sum(dim=0)
        return avg_predictions

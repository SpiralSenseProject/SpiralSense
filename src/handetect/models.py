#######################################################
# This file stores all the models used in the project.#
#######################################################

# Import all models from torchvision.models
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision.models import squeezenet1_0
from torchvision.models import vgg16
from torchvision.models import alexnet
from torchvision.models import densenet121
from torchvision.models import googlenet
from torchvision.models import inception_v3
from torchvision.models import mobilenet_v2
from torchvision.models import mobilenet_v3_small
from torchvision.models import mobilenet_v3_large
from torchvision.models import shufflenet_v2_x0_5
from torchvision.models import vgg11
from torchvision.models import vgg11_bn
from torchvision.models import vgg13
from torchvision.models import vgg13_bn
from torchvision.models import vgg16_bn
from torchvision.models import vgg19_bn
from torchvision.models import vgg19
from torchvision.models import wide_resnet50_2
from torchvision.models import wide_resnet101_2
from torchvision.models import mnasnet0_5
from torchvision.models import mnasnet0_75
from torchvision.models import mnasnet1_0
from torchvision.models import mnasnet1_3
from torchvision.models import resnext50_32x4d
from torchvision.models import resnext101_32x8d
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models import shufflenet_v2_x1_5
from torchvision.models import shufflenet_v2_x2_0
from torchvision.models import squeezenet1_1
from torchvision.models import efficientnet_v2_s
from torchvision.models import efficientnet_v2_m
from torchvision.models import efficientnet_v2_l
from torchvision.models import efficientnet_b0
from torchvision.models import efficientnet_b1
import torch
import torch.nn as nn

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


def ensemble_predictions(models, image):
    all_predictions = []

    with torch.no_grad():
        for model in models:
            output = model(image)
            all_predictions.append(output)

    return torch.stack(all_predictions, dim=0).mean(dim=0)


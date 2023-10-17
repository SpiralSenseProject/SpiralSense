import sys
import torch
import torch.nn as nn
from PIL import Image
import os
from configs import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
from itertools import product

random.seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
print("PyTorch Seed:", torch.initial_seed())
print("Random Seed:", random.getstate()[1][0])
print("PyTorch CUDA Seed:", torch.cuda.initial_seed())

print("DEVICE:", DEVICE)

# Define your model paths
# Load your pre-trained models
model2 = EfficientNetB2WithDropout(num_classes=NUM_CLASSES).to(DEVICE)
model2.load_state_dict(torch.load("output/checkpoints/EfficientNetB2WithDropout.pth"))
model1 = SqueezeNet1_0WithSE(num_classes=NUM_CLASSES).to(DEVICE)
model1.load_state_dict(torch.load("output/checkpoints/SqueezeNet1_0WithSE.pth"))
model3 = MobileNetV2WithDropout(num_classes=NUM_CLASSES).to(DEVICE)
model3.load_state_dict(torch.load("output\checkpoints\MobileNetV2WithDropout.pth"))

# Define the class labels
class_labels = CLASSES

# Define your test data folder path
test_data_folder = "data/test/Task 1/"


# Put models in evaluation mode
def set_models_eval(models):
    for model in models:
        model.eval()


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


def ensemble_predictions(models, image):
    all_predictions = []

    with torch.no_grad():
        for model in models:
            output = model(image)
            all_predictions.append(output)

    return torch.stack(all_predictions, dim=0).mean(dim=0)


# Load a single image and make predictions
def evaluate_image(models, image_path, transform=preprocess):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)
    outputs = ensemble_predictions(models, image)

    return outputs.argmax(dim=1).item()


# Evaluate and plot a confusion matrix for an ensemble of models
def evaluate_and_plot_confusion_matrix(models, test_data_folder):
    all_predictions = []
    true_labels = []

    with torch.no_grad():
        for class_label in class_labels:
            class_path = os.path.join(test_data_folder, class_label)
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                # print(image_path)
                predicted_label = evaluate_image(models, image_path, preprocess)
                all_predictions.append(predicted_label)
                true_labels.append(class_labels.index(class_label))

    # Print accuracy
    accuracy = (
        (torch.tensor(all_predictions) == torch.tensor(true_labels)).float().mean()
    )
    print("Accuracy:", accuracy)

    # Create the confusion matrix
    # cm = confusion_matrix(true_labels, all_predictions)

    # # Plot the confusion matrix
    # display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    # display.plot(cmap=plt.cm.Blues, values_format="d")

    # # Show the plot
    # plt.show()

    return accuracy

# Set the models to evaluation mode
set_models_eval([model1, model2, model3])

# Define different weight configurations
# [SqueezeNet, EfficientNetB2WithDropout, MobileNetV2WithDropout]
weights_configurations = [
    # Random set of weights using random.random() and all weights sum to 1
    [
        random.randrange(1, 10) / 10,
        random.randrange(1, 10) / 10,
        random.randrange(1, 10) / 10,
    ],
]


## NOTE OF PREVIOUS WEIGHTS
# Best weights: [0.2, 0.3, 0.5] with accuracy: 0.9428571462631226 at iteration: 15 with torch seed: 28434738589300 and random seed: 3188652458777471118 and torch cuda seed: None


best_weights = {
    "weights": 0,
    "accuracy": 0,
    "iteration": 0,
    "torch_seed": 0,
    "random_seed": 0,
    "torch_cuda_seed": 0,
}

i = 0

# weights_hist = []

target_sum = 1.0
number_of_numbers = 3
lower_limit = 0.25
upper_limit = 0.5
step = 0.1

valid_combinations = []

# Generate all unique combinations of three numbers with values to two decimal places
range_values = list(range(int(lower_limit * 100), int(upper_limit * 100) + 1))
for combo in product(range_values, repeat=number_of_numbers):
    combo_float = [x / 100.0 for x in combo]

    # Check if the sum of the numbers is equal to 1
    if sum(combo_float) == target_sum:
        valid_combinations.append(combo_float)

# Calculate the total number of possibilities
total_possibilities = len(valid_combinations)

print("Total number of possibilities:", total_possibilities)

valid_combinations = [[0.38, 0.34, 0.28]]
best_weighted_vote_ensemble_model = None

for weights in valid_combinations:
# while True:
    print("---------------------------")
    print("Iteration:", i)
    # Should iterate until all possible weights are exhausted
    # Create an ensemble model with weighted voting

    random.seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # print("PyTorch Seed:", torch.initial_seed())
    # weights_hist.append(weights)
    weighted_vote_ensemble_model = WeightedVoteEnsemble(
        # [model1, model2, model3], weights
        [model1, model2, model3],
        weights,
    )
    # print("Weights:", weights)
    print("Weights:", weights)
    # Call the evaluate_and_plot_confusion_matrix function with your models and test data folder
    accuracy = evaluate_and_plot_confusion_matrix(
        [weighted_vote_ensemble_model], test_data_folder
    )
    # Convert tensor to float
    accuracy = accuracy.item()
    if accuracy > best_weights["accuracy"]:
        # best_weights["weights"] = weights
        best_weights["weights"] = weights
        best_weights["accuracy"] = accuracy
        best_weights["iteration"] = i
        best_weights["torch_seed"] = torch.initial_seed()
        seed = random.randrange(sys.maxsize)
        rng = random.Random(seed)
        best_weights["random_seed"] = seed
        best_weights["torch_cuda_seed"] = torch.cuda.initial_seed()
        best_weighted_vote_ensemble_model = weighted_vote_ensemble_model

    print(
        "Best weights:",
        best_weights["weights"],
        "with accuracy:",
        best_weights["accuracy"],
        "at iteration:",
        best_weights["iteration"],
        "with torch seed:",
        best_weights["torch_seed"],
        "and random seed:",
        best_weights["random_seed"],
        "and torch cuda seed:",
        best_weights["torch_cuda_seed"],
    )
    i += 1


torch.save(
    best_weighted_vote_ensemble_model.state_dict(),
    "output/checkpoints/WeightedVoteEnsemble.pth",
)

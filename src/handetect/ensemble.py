import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from configs import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define your model paths
# Load your pre-trained models
model3 = MobileNetV2WithDropout(num_classes=NUM_CLASSES).to(DEVICE)
model3.load_state_dict(torch.load("output/checkpoints/MobileNetV2WithDropout.pth"))
model2 = EfficientNetB2WithDropout(num_classes=NUM_CLASSES).to(DEVICE)
model2.load_state_dict(torch.load("output/checkpoints/EfficientNetB2WithDropout.pth"))
model1 = SqueezeNet1_0WithSE(num_classes=NUM_CLASSES).to(DEVICE)
model1.load_state_dict(torch.load(r"output/checkpoints/SqueezeNet1_0WithSE.pth"))

models = [model1, model3, model2]

# Define the class labels
class_labels = CLASSES

# Define your test data folder path
test_data_folder = "data/test/Task 1/"


# Put models in evaluation mode
def set_models_eval(models):
    for model in models:
        model.eval()


# Define the ensemble model using a list of models
class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        predictions = [model(x) for model in self.models]
        avg_predictions = torch.stack(predictions, dim=0).mean(dim=0)
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
                predicted_label = evaluate_image(models, image_path, preprocess)
                all_predictions.append(predicted_label)
                true_labels.append(class_labels.index(class_label))

    # Print accuracy
    accuracy = (torch.tensor(all_predictions) == torch.tensor(true_labels)).float().mean()
    print("Accuracy:", accuracy)

    # Create the confusion matrix
    cm = confusion_matrix(true_labels, all_predictions)

    # Plot the confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_CLASSES))
    display.plot(cmap=plt.cm.Blues, values_format="d")

    # Show the plot
    plt.show()


# Set the models to evaluation mode
set_models_eval(models)

# Create an ensemble model
ensemble_model = EnsembleModel(models)

# Call the evaluate_and_plot_confusion_matrix function with your models and test data folder
evaluate_and_plot_confusion_matrix([ensemble_model], test_data_folder)

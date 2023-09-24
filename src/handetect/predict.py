import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from handetect.models import *
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt

# Define the path to your model checkpoint
model_checkpoint_path = "model.pth"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 6

# Define transformation for preprocessing the input image
preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize the image to match training input size
        transforms.Grayscale(num_output_channels=3),  # Convert the image to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image
    ]
)

# Load your model (change this according to your model definition)
model = vgg16(pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(
    torch.load(model_checkpoint_path, map_location=DEVICE)
)  # Load the model on the same device
model.eval()
model = model.to(DEVICE)
model.eval()
torch.set_grad_enabled(False)


def predict_image(image_path, model=model, transform=preprocess):
    # Define images variable to recursively list all the data file in the image_path
    classes = ['Cerebral Palsy', 'Dystonia', 'Essential Tremor', 'Healthy', 'Huntington Disease', 'Parkinson Disease']

    print("---------------------------")
    print("Image path:", image_path)
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)
    output = model(image)

    # softmax algorithm
    probabilities = torch.softmax(output, dim=1)[0] * 100

    # Sort the classes by probabilities in descending order
    sorted_classes = sorted(
        zip(classes, probabilities), key=lambda x: x[1], reverse=True
    )

    # Report the prediction for each class
    print("Probabilities for each class:")
    for class_label, class_prob in sorted_classes:
        class_prob = class_prob.item().__round__(2)
        print(f"{class_label}: {class_prob}%")

    # Get the predicted class
    predicted_class = sorted_classes[0][0]  # Most probable class
    predicted_label = classes.index(predicted_class)

    # Report the prediction
    print("Predicted class:", predicted_label)
    print("Predicted label:", predicted_class)
    print("---------------------------")

    return sorted_classes


# # Call the predict_image function
# predicted_label, sorted_probabilities = predict_image(image_path, model, preprocess)

# # Access probabilities for each class in sorted order
# for class_label, class_prob in sorted_probabilities:
#     print(f"{class_label}: {class_prob}%")

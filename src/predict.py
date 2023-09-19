import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import ImageReadMode
from PIL import Image
from models import *

# Define the path to your model checkpoint
model_checkpoint_path = "model.pth"

# Define the path to the image you want to classify
image_path = r"data\train\Task 1\Dystonia\09.png"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3


# Create a parameter for the number of classes to be entered by user
def __init__(self, num_classes):
    

# model = ResNet50(num_classes=3)
model = resnet18(pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()
model=model.to(DEVICE)
# Define transformation for preprocessing the input image
preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize the image to match training input size
        transforms.Grayscale(num_output_channels=3),  # Convert the image to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image
    ]
)

# Load and preprocess the image
image = Image.open(image_path)
image = preprocess(image)
image = image.unsqueeze(0)  # Add a batch dimension to the input tensor
image=image.to(DEVICE)
# Make predictions
with torch.no_grad():
    output = model(image)
    _, predicted_class = torch.max(output, 1)

# Print the predicted class
print(f"Predicted class = {predicted_class.item()}")

# Define the class labels
class_labels = ['Dystonia', 'Essential Tremor', 'Parkinson Disease']
                
def predict_image(image_path, model, transform, class_labels):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(image)

    # softmax algorithm
    probabilities = torch.softmax(output, dim=1)[0] * 100
    predicted_class = torch.argmax(output, dim=1).item()

    # Print predicted class and probablity
    print("Predicted class:", class_labels[predicted_class])
    print("Probability:", probabilities[predicted_class].item())


predict_image(image_path, model, preprocess, class_labels)
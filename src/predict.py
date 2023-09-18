import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import *

# Define the path to your model checkpoint
model_checkpoint_path = "model.pth"

# Define the path to the image you want to classify
image_path = r"17flowers/test/1/image_0240.jpg"

RANDOM_SEED = 123
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
STEP_SIZE = 10
GAMMA = 0.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_PRINT = 100


model = VGG16(num_classes=3)
model.load_state_dict(torch.load(model_checkpoint_path))
model.eval()
model=model.to(DEVICE)
# Define transformation for preprocessing the input image
preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize the image to match training input size
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

class_labels = ["Daaffodil", "LilyValley", "Snowdrop"]

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
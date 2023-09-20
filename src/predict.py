import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import *  # Make sure you import your model correctly from the 'models' module
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import pathlib 

# Define the path to your model checkpoint
model_checkpoint_path = "model.pth"

# Define the path to the image you want to classify
image_path = "data/test/Task 1/"  # Use forward slashes for file paths

# Define images variable to recursively list all the data file in the image_path
images = list(pathlib.Path(image_path).rglob("*.png"))
classes = os.listdir(image_path)
print(images)

true_classs = []
predicted_labels = []

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 5  # Update with the correct number of classes

# Load your model (change this according to your model definition)
model = resnet18(pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE))  # Load the model on the same device
model.eval()
model = model.to(DEVICE)

# Define transformation for preprocessing the input image
preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize the image to match training input size
        transforms.Grayscale(num_output_channels=3),  # Convert the image to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image
    ]
)

def predict_image(image_path, model, transform):
    model.eval()
    correct_predictions = 0
    total_predictions = len(images)
    
    with torch.no_grad():
        for i in images:
            print('---------------------------')
            # Check the true label of the image by checking the sequence of the folder in Task 1
            true_class = classes.index(i.parts[-2])
            print("Image path:", i)
            print("True class:", true_class)
            image = Image.open(i)
            image = transform(image).unsqueeze(0)
            image = image.to(DEVICE)
            output = model(image)

            # softmax algorithm
            probabilities = torch.softmax(output, dim=1)[0] * 100
            predicted_class = torch.argmax(output, dim=1).item()

            # Append true and predicted labels to their respective lists
            true_classs.append(true_class)
            predicted_labels.append(predicted_class)
            
            # Check if the prediction is correct
            if predicted_class == true_class:
                correct_predictions += 1

            # Report the prediction
            print("Predicted class:", predicted_class)
            print("Probability:", probabilities[predicted_class].item())
            print("Predicted label:", classes[predicted_class])
            print("Correct predictions:", correct_predictions)
            print("Correct?", "Yes" if predicted_class == true_class else "No")
            print("---------------------------")
            


    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)

# Call the predict_image function
predict_image(image_path, model, preprocess)

# Convert the lists to tensors
predicted_labels_tensor = torch.tensor(predicted_labels)
true_classs_tensor = torch.tensor(true_classs)

# Create confusion matrix
conf_matrix = ConfusionMatrix(num_classes=NUM_CLASSES, task='multiclass')
conf_matrix.update(predicted_labels_tensor, true_classs_tensor)

# Plot confusion matrix
conf_matrix.plot()
plt.show()

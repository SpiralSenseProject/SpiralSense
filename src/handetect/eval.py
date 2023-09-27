import os
import torch
from torchvision.transforms import transforms
from sklearn.metrics import f1_score
import pathlib
from PIL import Image
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
from configs import *
from data_loader import load_data  # Import the load_data function

image_path = "data/test/Task 1/"

# Constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model
MODEL = MODEL.to(DEVICE)
MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
MODEL.eval()


def predict_image(image_path, model, transform):
    model.eval()
    correct_predictions = 0

    # Get a list of image files
    images = list(pathlib.Path(image_path).rglob("*.png"))

    total_predictions = len(images)

    true_classes = []
    predicted_labels = []

    with torch.no_grad():
        for image_file in images:
            print("---------------------------")
            # Check the true label of the image by checking the sequence of the folder in Task 1
            true_class = CLASSES.index(image_file.parts[-2])
            print("Image path:", image_file)
            print("True class:", true_class)
            image = Image.open(image_file).convert("RGB")
            image = transform(image).unsqueeze(0)
            image = image.to(DEVICE)
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()
            # Print the predicted class
            print("Predicted class:", predicted_class)
            # Append true and predicted labels to their respective lists
            true_classes.append(true_class)
            predicted_labels.append(predicted_class)

            # Check if the prediction is correct
            if predicted_class == true_class:
                correct_predictions += 1

    # Calculate accuracy and f1 score
    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)
    f1 = f1_score(true_classes, predicted_labels, average="weighted")
    print("Weighted F1 Score:", f1)

    # Convert the lists to tensors
    predicted_labels_tensor = torch.tensor(predicted_labels)
    true_classes_tensor = torch.tensor(true_classes)

    # Create a confusion matrix
    conf_matrix = ConfusionMatrix(num_classes=NUM_CLASSES, task="multiclass")
    conf_matrix.update(predicted_labels_tensor, true_classes_tensor)

    # Plot the confusion matrix
    conf_matrix.compute()
    conf_matrix.plot()
    plt.show()


# Call predict_image function
predict_image(image_path, MODEL, preprocess)

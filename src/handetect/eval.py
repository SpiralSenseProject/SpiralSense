import os
import torch
import numpy as np
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from configs import *
from data_loader import load_data  # Import the load_data function

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
    predicted_scores = []  # To store predicted class probabilities

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
            predicted_scores.append(
                output.softmax(dim=1).cpu().numpy()
            )  # Store predicted class probabilities

            # Check if the prediction is correct
            if predicted_class == true_class:
                correct_predictions += 1

    # Calculate accuracy and f1 score
    accuracy = accuracy_score(true_classes, predicted_labels)
    print("Accuracy:", accuracy)
    f1 = f1_score(true_classes, predicted_labels, average="weighted")
    print("Weighted F1 Score:", f1)

    # Convert the lists to tensors
    predicted_labels_tensor = torch.tensor(predicted_labels)
    true_classes_tensor = torch.tensor(true_classes)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_labels)

    # Plot the confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=CLASSES).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Classification report
    class_names = [str(cls) for cls in range(NUM_CLASSES)]
    report = classification_report(
        true_classes, predicted_labels, target_names=class_names
    )
    print("Classification Report:\n", report)

    # Calculate precision and recall for each class
    true_classes_binary = label_binarize(true_classes, classes=range(NUM_CLASSES))
    precision, recall, _ = precision_recall_curve(
        true_classes_binary.ravel(), np.array(predicted_scores).ravel()
    )

    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


# Call predict_image function with your image path
predict_image("data/test/Task 1/", MODEL, preprocess)

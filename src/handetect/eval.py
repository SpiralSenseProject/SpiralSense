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
)
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from configs import *

# Constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_AUGMENTATIONS = 10  # Number of augmentations to perform

# Load the model
model = MODEL.to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# define augmentations for TTA
tta_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]
)


def perform_tta(model, image, tta_transforms):
    augmented_predictions = []
    augmented_scores = []

    for _ in range(NUM_AUGMENTATIONS):
        augmented_image = tta_transforms(image)
        output = model(augmented_image)
        predicted_class = torch.argmax(output, dim=1).item()
        augmented_predictions.append(predicted_class)
        augmented_scores.append(output.softmax(dim=1).cpu().numpy())

    # max voting
    final_predicted_class_max = max(
        set(augmented_predictions), key=augmented_predictions.count
    )

    # average probabilities
    final_predicted_scores_avg = np.mean(np.array(augmented_scores), axis=0)

    # rotate and average probabilities
    rotation_transforms = [
        transforms.RandomRotation(degrees=i) for i in range(0, 360, 30)
    ]
    rotated_scores = []
    for rotation_transform in rotation_transforms:
        augmented_image = rotation_transform(image)
        output = model(augmented_image)
        rotated_scores.append(output.softmax(dim=1).cpu().numpy())

    final_predicted_scores_rotation = np.mean(np.array(rotated_scores), axis=0)

    return (
        final_predicted_class_max,
        final_predicted_scores_avg,
        final_predicted_scores_rotation,
    )


def predict_image_with_tta(image_path, model, transform, tta_transforms):
    model.eval()
    correct_predictions = 0
    true_classes = []
    predicted_labels_max = []
    predicted_labels_avg = []
    predicted_labels_rotation = []

    with torch.no_grad():
        images = list(pathlib.Path(image_path).rglob("*.png"))
        total_predictions = len(images)

        for image_file in images:
            true_class = CLASSES.index(image_file.parts[-2])

            original_image = Image.open(image_file).convert("RGB")
            original_image = transform(original_image).unsqueeze(0)
            original_image = original_image.to(DEVICE)

            # Perform TTA with different strategies
            final_predicted_class_max, _, _ = perform_tta(
                model, original_image, tta_transforms
            )
            _, final_predicted_scores_avg, _ = perform_tta(
                model, original_image, tta_transforms
            )
            _, _, final_predicted_scores_rotation = perform_tta(
                model, original_image, tta_transforms
            )

            true_classes.append(true_class)
            predicted_labels_max.append(final_predicted_class_max)
            predicted_labels_avg.append(np.argmax(final_predicted_scores_avg))
            predicted_labels_rotation.append(np.argmax(final_predicted_scores_rotation))

            if final_predicted_class_max == true_class:
                correct_predictions += 1

    # accuracy for each strategy
    accuracy_max = accuracy_score(true_classes, predicted_labels_max)
    accuracy_avg = accuracy_score(true_classes, predicted_labels_avg)
    accuracy_rotation = accuracy_score(true_classes, predicted_labels_rotation)

    print("Accuracy (Max Voting):", accuracy_max)
    print("Accuracy (Average Probabilities):", accuracy_avg)
    print("Accuracy (Rotation and Average):", accuracy_rotation)

    # final prediction using ensemble (choose the strategy with the highest accuracy)
    final_predicted_labels = []
    for i in range(len(true_classes)):
        max_strategy_accuracy = max(accuracy_max, accuracy_avg, accuracy_rotation)
        if accuracy_max == max_strategy_accuracy:
            final_predicted_labels.append(predicted_labels_max[i])
        elif accuracy_avg == max_strategy_accuracy:
            final_predicted_labels.append(predicted_labels_avg[i])
        else:
            final_predicted_labels.append(predicted_labels_rotation[i])

    # calculate accuracy and f1 score(ensemble)
    accuracy_ensemble = accuracy_score(true_classes, final_predicted_labels)
    f1_ensemble = f1_score(true_classes, final_predicted_labels, average="weighted")

    print("Ensemble Accuracy:", accuracy_ensemble)
    print("Ensemble Weighted F1 Score:", f1_ensemble)

    # Classification report
    class_names = [str(cls) for cls in range(NUM_CLASSES)]
    report = classification_report(
        true_classes, final_predicted_labels, target_names=class_names
    )
    print("Classification Report of", MODEL.__class__.__name__, ":\n", report)
    
    # confusion matrix and classification report for the ensemble
    conf_matrix_ensemble = confusion_matrix(true_classes, final_predicted_labels)
    ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix_ensemble, display_labels=range(NUM_CLASSES)
    ).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Ensemble)")
    plt.show()

    class_names = [str(cls) for cls in range(NUM_CLASSES)]
    report_ensemble = classification_report(
        true_classes, final_predicted_labels, target_names=class_names
    )
    print("Classification Report (Ensemble):\n", report_ensemble)

    # Calculate precision and recall for each class
    true_classes_binary = label_binarize(true_classes, classes=range(NUM_CLASSES))
    precision, recall, _ = precision_recall_curve(
        true_classes_binary.ravel(), np.array(final_predicted_scores_rotation).ravel()
    )

    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

predict_image_with_tta("data/test/Task 1/", model, preprocess, tta_transforms)

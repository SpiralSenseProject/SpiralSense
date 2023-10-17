import os
import torchvision
import shap
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
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from configs import *
from data_loader import load_data  # Import the load_data function

# MobileNet: 0.8731189445475158
# EfficientNet: 0.873118944547516
# SquuezeNet: 0.8865856365856365


# Load the model
# MODEL = MODEL.to(DEVICE)
# MODEL.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
# MODEL.eval()

model2 = EfficientNetB2WithDropout(num_classes=NUM_CLASSES).to(DEVICE)
model2.load_state_dict(torch.load("output/checkpoints/EfficientNetB2WithDropout.pth"))
model1 = SqueezeNet1_0WithSE(num_classes=NUM_CLASSES).to(DEVICE)
model1.load_state_dict(torch.load("output/checkpoints/SqueezeNet1_0WithSE.pth"))
model3 = MobileNetV2WithDropout(num_classes=NUM_CLASSES).to(DEVICE)
model3.load_state_dict(torch.load("output\checkpoints\MobileNetV2WithDropout.pth"))

model1.eval()
model2.eval()
model3.eval()

# Load the model
model = WeightedVoteEnsemble([model1, model2, model3], [0.38, 0.34, 0.28])
# model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.load_state_dict(
    torch.load("output/checkpoints/WeightedVoteEnsemble.pth", map_location=DEVICE)
)
model.eval()


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
    ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=range(NUM_CLASSES)
    ).plot(cmap=plt.cm.Blues)
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

    fpr, tpr, _ = roc_curve(
        true_classes_binary.ravel(), np.array(predicted_scores).ravel()
    )
    roc_auc = auc(fpr, tpr)
    print("ROC AUC:", roc_auc)

    # Calculate PRC AUC
    precision, recall, _ = precision_recall_curve(
        true_classes_binary.ravel(), np.array(predicted_scores).ravel()
    )
    prc_auc = average_precision_score(
        true_classes_binary.ravel(), np.array(predicted_scores).ravel()
    )
    print("PRC AUC:", prc_auc)

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


predict_image("data/test/Task 1/", model, preprocess)


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img))
    img = img.to(DEVICE)
    output = model(img)
    return output


X, y = load_data("data/test/Task 1/", CLASSES)

# Check that transformations work correctly
Xtr = preprocess(torch.Tensor(X))
out = predict(Xtr[1:3])
classes = torch.argmax(out, axis=1).cpu().numpy()
print(f"Classes: {classes}: {np.array(CLASSES)[classes]}")

topk = 4
batch_size = 50
n_evals = 10000

# define a masker that is used to mask out partitions of the input image.
masker_blur = shap.maskers.Image("blur(128,128)", Xtr[0].shape)

# create an explainer with model and image masker
explainer = shap.Explainer(predict, masker_blur, output_names=CLASSES)

# feed only one image
# here we explain two images using 100 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer(
    Xtr[1:2],
    max_evals=n_evals,
    batch_size=batch_size,
    outputs=shap.Explanation.argsort.flip[:topk],
)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

inv_transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist(),
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

inv_transform = torchvision.transforms.Compose(inv_transform)

shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

shap.image_plot(
    shap_values=shap_values.values,
    pixel_values=shap_values.data,
    labels=shap_values.output_names,
    true_labels=[CLASSES[132]],
)
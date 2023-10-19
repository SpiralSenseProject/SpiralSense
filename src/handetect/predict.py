import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import *
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
from configs import *


# Load your model (change this according to your model definition)
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
model = MODEL.to(DEVICE)
# model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.load_state_dict(
    torch.load("output/checkpoints/EfficientNetB3WithDropout.pth", map_location=DEVICE)
)
model.eval()

torch.set_grad_enabled(False)


def predict_image(image_path, model=model, transform=preprocess):
    classes = CLASSES

    print("---------------------------")
    print("Image path:", image_path)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(DEVICE)
    output = model(image)

    # Softmax algorithm
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

    return sorted_classes

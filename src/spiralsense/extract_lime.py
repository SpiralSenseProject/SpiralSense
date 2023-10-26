import os
import numpy as np
from lime.lime_image import LimeImageExplainer
from PIL import Image
import torch
import matplotlib.pyplot as plt
from configs import *


model = MODEL.to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()


# Define a function to predict with the model
def predict(input_image):
    input_image = torch.tensor(input_image, dtype=torch.float32)
    if input_image.dim() == 4:
        input_image = input_image.permute(0, 3, 1, 2)  # Permute the dimensions
    input_image = input_image.to(DEVICE)  # Move to the appropriate device
    with torch.no_grad():
        output = model(input_image)
    return output


def generate_lime(image_path=None, save_path=None):
    if image_path is None:
        for disease in CLASSES:
            print("Processing", disease)
            for image_path in os.listdir(r"data\test\Task 1\{}".format(disease)):
                image = None
                print("Processing", image_path)
                image_path = r"data\test\Task 1\{}\{}".format(disease, image_path)
                image_name = image_path.split(".")[0].split("\\")[-1]
                image = Image.open(image_path).convert("RGB")
                image = preprocess(image)
                image = image.unsqueeze(0)  # Add batch dimension
                image = image.to(DEVICE)

                # Create the LIME explainer
                explainer = LimeImageExplainer()

                # Explain the model's predictions for the image
                explanation = explainer.explain_instance(
                    image[0].permute(1, 2, 0).numpy(),
                    predict,
                    top_labels=5,
                    num_samples=1000,
                )

                # Get the image and mask for the explanation
                image, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=False,
                    num_features=10,
                    hide_rest=False,
                )

                # Save the image (dun use plt.imsave)
                # Normalize the image to the [0, 1] range
                # norm = Normalize(vmin=0, vmax=1)
                # image = norm(image)

                image = (image - np.min(image)) / (np.max(image) - np.min(image))

                # image = Image.fromarray(image)
                os.makedirs(f"docs/evaluation/lime/{disease}", exist_ok=True)
                # image.save(f'docs/evaluation/lime/{disease}/{image_name}.jpg')
                plt.imsave(f"docs/evaluation/lime/{disease}/{image_name}.jpg", image)

    else:
        image = None
        print("Processing", image_path)
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(DEVICE)

        # Create the LIME explainer
        explainer = LimeImageExplainer()

        # Explain the model's predictions for the image
        explanation = explainer.explain_instance(
            image[0].permute(1, 2, 0).numpy(), predict, top_labels=5, num_samples=1000
        )

        # Get the image and mask for the explanation
        image, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )

        # Save the image (dun use plt.imsave)
        # Normalize the image to the [0, 1] range
        # norm = Normalize(vmin=0, vmax=1)
        # image = norm(image)

        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # image = Image.fromarray(image)
        # os.makedirs(f"docs/evaluation/lime/{disease}", exist_ok=True)
        # image.save(f'docs/evaluation/lime/{disease}/{image_name}.jpg')
        plt.imsave(save_path, image)


generate_lime()

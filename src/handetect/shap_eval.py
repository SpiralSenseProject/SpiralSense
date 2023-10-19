import numpy as np
from lime.lime_image import LimeImageExplainer
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from configs import *


model = MODEL.to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# Load the image
image = Image.open(
    r"data\test\Task 1\Healthy\0a7259b2-e650-43aa-93a0-e8b1063476fc.png"
).convert("RGB")
image = preprocess(image)
image = image.unsqueeze(0)  # Add batch dimension
image = image.to(DEVICE)


# Define a function to predict with the model
def predict(input_image):
    input_image = torch.tensor(input_image, dtype=torch.float32)
    if input_image.dim() == 4:
        input_image = input_image.permute(0, 3, 1, 2)  # Permute the dimensions
    input_image = input_image.to(DEVICE)  # Move to the appropriate device
    with torch.no_grad():
        output = model(input_image)
    return output


# Create the LIME explainer
explainer = LimeImageExplainer()

# Explain the model's predictions for the image
explanation = explainer.explain_instance(
    image[0].permute(1, 2, 0).numpy(), predict, top_labels=5, num_samples=2000
)

# Get the image and mask for the explanation
image, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
)

# Display the explanation
plt.imshow(image)
plt.show()

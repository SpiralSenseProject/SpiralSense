from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np
import torch
import torch.nn as nn  # Replace with your model
from configs import *
import os, random

# Load your model (replace with your model class)
model = MODEL.to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# Find the target layer (modify this based on your model architecture)
target_layer = None
for child in model.features[-1]:
    if isinstance(child, nn.Conv2d):
        target_layer = child

if target_layer is None:
    raise ValueError("Invalid layer name: {}".format(target_layer))

def extract_gradcam(image_path=None, save_path=None):
    if image_path is None:
        for disease in CLASSES:
            print("Processing", disease)
            image_path = random.choice(os.listdir("data/test/Task 1/" + disease))
            image_path = "data/test/Task 1/" + disease + "/" + image_path
            rgb_img = cv2.imread(image_path, 1)
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            input_tensor = input_tensor.to(DEVICE)

            # Create a GradCAMPlusPlus object
            cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=True)

            # Generate the GradCAM heatmap
            grayscale_cam = cam(input_tensor=input_tensor)[0]

            # Apply a colormap to the grayscale heatmap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)

            # Ensure heatmap_colored has the same dtype as rgb_img
            heatmap_colored = heatmap_colored.astype(np.float32) / 255

            # Adjust the alpha value to control transparency
            alpha = 0.3  # You can change this value to make the original image more or less transparent

            # Overlay the colored heatmap on the original image
            final_output = cv2.addWeighted(rgb_img, 0.3, heatmap_colored, 0.7, 0)

            # Save the final output
            cv2.imwrite(f'docs/efficientnet/gradcam/{disease}.jpg', (final_output * 255).astype(np.uint8))
    else:
            rgb_img = cv2.imread(image_path, 1)
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            input_tensor = input_tensor.to(DEVICE)

            # Create a GradCAMPlusPlus object
            cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

            # Generate the GradCAM heatmap
            grayscale_cam = cam(input_tensor=input_tensor)[0]

            # Apply a colormap to the grayscale heatmap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)

            # Ensure heatmap_colored has the same dtype as rgb_img
            heatmap_colored = heatmap_colored.astype(np.float32) / 255

            # Adjust the alpha value to control transparency
            alpha = 0.3  # You can change this value to make the original image more or less transparent

            # Overlay the colored heatmap on the original image
            final_output = cv2.addWeighted(rgb_img, 0.3, heatmap_colored, 0.7, 0)

            # Save the final output
            cv2.imwrite(save_path, (final_output * 255).astype(np.uint8))
            
            return save_path            
            

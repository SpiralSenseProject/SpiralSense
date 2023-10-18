from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np
import torch
import torch.nn as nn  # Replace with your model
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


# Find the target layer (modify this based on your model architecture)
# EfficientNetB2WithDropout - model.features[-1]
# SqueezeNet1_0WithSE - model.features
# MobileNetV2WithDropout - model.features[-1]

target_layer_efficientnet = None
for child in model2.features[-1]:
    if isinstance(child, nn.Conv2d):
        target_layer_efficientnet = child

if target_layer_efficientnet is None:
    raise ValueError(
        "Invalid EfficientNet layer name: {}".format(target_layer_efficientnet)
    )

target_layer_squeezenet = None
for child in model1.features:
    if isinstance(child, nn.Conv2d):
        target_layer_squeezenet = child

if target_layer_squeezenet is None:
    raise ValueError(
        "Invalid SqueezeNet layer name: {}".format(target_layer_squeezenet)
    )

target_layer_mobilenet = None
for child in model3.features[-1]:
    if isinstance(child, nn.Conv2d):
        target_layer_mobilenet = child

if target_layer_mobilenet is None:
    raise ValueError("Invalid MobileNet layer name: {}".format(target_layer_mobilenet))

# Load and preprocess the image
image_path = r"data\test\Task 1\Cerebral Palsy\89.png"
rgb_img = cv2.imread(image_path, 1)
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
input_tensor = input_tensor.to(DEVICE)
input_tensor.requires_grad = True  # Enable gradients for the input tensor


# Create a GradCAMPlusPlus object
efficientnet_cam = GradCAMPlusPlus(model=model2, target_layers=[target_layer_efficientnet], use_cuda=True)
squeezenet_cam = GradCAMPlusPlus(model=model1, target_layers=[target_layer_squeezenet], use_cuda=True)
mobilenet_cam = GradCAMPlusPlus(model=model3, target_layers=[target_layer_mobilenet], use_cuda=True)


efficientnet_grayscale_cam = efficientnet_cam(input_tensor=input_tensor)[0]
squeezenet_grayscale_cam = squeezenet_cam(input_tensor=input_tensor)[0]
mobilenet_grayscale_cam = mobilenet_cam(input_tensor=input_tensor)[0]

# Apply a colormap to the grayscale heatmap
efficientnet_heatmap_colored = cv2.applyColorMap(np.uint8(255 * efficientnet_grayscale_cam), cv2.COLORMAP_JET)
squeezenet_heatmap_colored = cv2.applyColorMap(np.uint8(255 * squeezenet_grayscale_cam), cv2.COLORMAP_JET)
mobilenet_heatmap_colored = cv2.applyColorMap(np.uint8(255 * mobilenet_grayscale_cam), cv2.COLORMAP_JET)

# normalized_efficientnet_heatmap = efficientnet_heatmap_colored / np.max(efficientnet_heatmap_colored)
# normalized_squeezenet_heatmap = squeezenet_heatmap_colored / np.max(squeezenet_heatmap_colored)
# normalized_mobilenet_heatmap = mobilenet_heatmap_colored / np.max(mobilenet_heatmap_colored)

# # Ensure heatmap_colored has the same dtype as rgb_img
# normalized_efficientnet_heatmap = normalized_efficientnet_heatmap.astype(np.float32) / 255
# normalized_squeezenet_heatmap = normalized_squeezenet_heatmap.astype(np.float32) / 255
# normalized_mobilenet_heatmap = normalized_mobilenet_heatmap.astype(np.float32) / 255

efficientnet_heatmap_colored = efficientnet_heatmap_colored.astype(np.float32) / 255
squeezenet_heatmap_colored = squeezenet_heatmap_colored.astype(np.float32) / 255
mobilenet_heatmap_colored = mobilenet_heatmap_colored.astype(np.float32) / 255

# Adjust the alpha value to control transparency
alpha = (
    0.1  # You can change this value to make the original image more or less transparent
)


# [0.38, 0.34, 0.28]
weighted_heatmap = (
    efficientnet_heatmap_colored * 0.38
    + squeezenet_heatmap_colored * 0.34
    + mobilenet_heatmap_colored * 0.28
)


# Overlay the colored heatmap on the original image
final_output = cv2.addWeighted(rgb_img, 0.3, weighted_heatmap, 0.7, 0)

# Save the final output
cv2.imwrite("cam.jpg", (final_output * 255).astype(np.uint8))

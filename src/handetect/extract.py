from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
import cv2
import numpy as np
import os
from configs import *

DEVICE = torch.device("cpu")
# clear cuda cache
torch.cuda.empty_cache()

model = MODEL.to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# target_layer = model.layer4[-1]
def find_target_layer(model, target_layer_name):
    for name, layer in model.named_modules():
        if name == target_layer_name:
            return layer
    return None

# Assuming 'model' is an instance of EfficientNetB2WithDropout
target_layer = None

target_layer = None
for child in model.features[-1]:
    if isinstance(child, nn.Conv2d):
        target_layer = child
# Verify that you have the correct layer
print(target_layer)

#Resnet18 and 50: model.layer4[-1]
#VGG and densenet161: model.features[-1]
#mnasnet1_0: model.layers[-1]
#ViT: model.blocks[-1].norm1
#SqueezeNet1_0: model.features[-1]

image_path = r'data\train\external\Task 1\Essential Tremor\07.png'
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]   
                                                 
rgb_img = cv2.imread(image_path, 1) 
rgb_img = np.float32(rgb_img) / 255



input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])   # torch.Size([1, 3, 224, 224])
input_tensor = input_tensor.to(DEVICE)
# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=True)

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor)  # [batch, 224,224]

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0]
visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)
cv2.imwrite('cam.jpg', visualization) 
import torch
from torchvision.models import vgg16, VGG16_Weights
from src.lrp import LRPModel
from configs import *
from PIL import Image


image = Image.open(r'data\test\Task 1\Alzheimer Disease\0d846ee1-c90d-4ed5-8467-3550dd653858.png').convert("RGB")
image = preprocess(image).unsqueeze(0)
image = image.to(DEVICE)
model = MODEL.to(DEVICE)
print(dict(model.named_modules()))
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()
lrp_model = LRPModel(model)
r = lrp_model.forward(image)

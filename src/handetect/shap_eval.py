import numpy as np
import torch
import torchvision
from configs import *
import shap
from PIL import Image

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

classes_name = CLASSES

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


def nhwc_to_nchw(x: torch.Tensor):
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor):
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


inv_transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist(),
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

transform = preprocess
inv_transform = torchvision.transforms.Compose(inv_transform)
image_file = r"data\train\external\Task 1\Essential Tremor\03.png"


def predict(img: np.ndarray):
    # img = nhwc_to_nchw(torch.Tensor(img))
    img = img.to(DEVICE)
    output = model(img)
    return output


# Check that transformations work correctly
Xtr = Image.open(image_file).convert("RGB")
Xtr = transform(Xtr).unsqueeze(0)
Xtr = Xtr.to(DEVICE)
out = predict(Xtr)
classes = torch.argmax(out, axis=1).cpu().numpy()
print(f"Classes: {classes}: {np.array(classes_name)[classes]}")
import numpy as np
import torch
import torchvision
from configs import *
import shap
from PIL import Image

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

classes_name = CLASSES

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


def nhwc_to_nchw(x: torch.Tensor):
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor):
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


inv_transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(
        mean=[-1 * m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]


def inverse_preprocess(img: torch.Tensor):
    img = img.to(DEVICE)  # Make sure it's on the appropriate device
    img = inv_transform[0](img)
    img = inv_transform[1](img)
    img = inv_transform[2](img)
    return img


# Load and preprocess the image
image_file = "data/train/external/Task 1/Essential Tremor/03.png"
Xtr = Image.open(image_file).convert("RGB")
Xtr = preprocess(Xtr).unsqueeze(0).to(DEVICE)


# Define a function to make predictions
def predict(img):
    output = ensemble_predictions([model1, model2, model3], img)
    return output


# Get the model's predictions and class labels
out = predict(Xtr)
classes = torch.argmax(out, axis=1).cpu().numpy()
class_names = np.array(classes_name)[classes]
print(f"Predicted Class: {classes}: {class_names}")

Xtr = Xtr.to(DEVICE)

# Set up SHAP explanations
topk = 4
batch_size = 1
n_evals = 10000

# Define a masker to use with SHAP
masker_blur = shap.maskers.Image(mask_value="inpaint_telea", shape=Xtr[0].shape)

# Create an explainer for the model
explainer = shap.Explainer(predict, masker_blur, output_names=classes_name)

# Generate SHAP values
shap_values = explainer(
    Xtr[0],
    max_evals=n_evals,
    batch_size=batch_size,
    outputs=shap.Explanation.argsort.flip[:topk],
)

# Inverse transform the SHAP values and pixel values
shap_values.data = inverse_preprocess(shap_values.data).cpu().numpy()[0]
shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

# Visualize SHAP values
shap.image_plot(
    shap_values=shap_values.values,
    pixel_values=shap_values.data,
    labels=shap_values.output_names,
    true_labels=[class_names],
)

import torch
from torchvision import models, transforms
from PIL import Image

# Define a function to load the model
def load_model(model_path):
    model = models.resnet18(pretrained=True)  # Load a pre-trained ResNet-18 model as an example
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 6)  # Replace the last fully connected layer for your specific task
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define a function to preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.CenterCrop((1680, 940)),
        transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension
    return image

# Define the path to your trained model
model_path = 'model.pth'  # Replace with the actual path to your model file

# Load the model
model = load_model(model_path)

# Define the path to the image you want to classify
image_path = 'data/test/Task 1/test_cerebral_palsy.png'  # Replace with the actual path to your image file
# image_path = 'test_cerebral_palsy.png'  # Replace with the actual path to your image file

# Preprocess the image
input_image = preprocess_image(image_path)

# Make predictions
with torch.no_grad():
    output = model(input_image)

# Assuming this is a classification task, get the predicted class index
predicted_class = output.max(1)[1]

# The variable 'predicted_class' contains the predicted class index
print("Predicted class:", predicted_class.item())

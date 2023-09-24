import os
import torch
from torchvision.transforms import transforms
from sklearn.metrics import f1_score
from models import * 
import pathlib
from PIL import Image
from torchmetrics import ConfusionMatrix, Accuracy
import matplotlib.pyplot as plt

model_checkpoint_path = "model.pth" 
image_path = "data/test/Task 1/" 

# constants
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6 

# load the model
images = list(pathlib.Path(image_path).rglob("*.png"))
classes = os.listdir(image_path)
print(images)

true_classs = []
predicted_labels = []

model = vgg16(pretrained=False, num_classes=NUM_CLASSES)  
model.load_state_dict(torch.load(model_checkpoint_path, map_location=DEVICE)) 
model.eval()
model = model.to(DEVICE)


# Define transformation for preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize (for grayscale)
    ]
)

# evaluate the model
all_predictions = []
true_labels = []

def predict_image(image_path, model, transform):
    model.eval()
    correct_predictions = 0
    total_predictions = len(images)
    
    with torch.no_grad():
        for i in images:
            print('---------------------------')
            # Check the true label of the image by checking the sequence of the folder in Task 1
            true_class = classes.index(i.parts[-2])
            print("Image path:", i)
            print("True class:", true_class)
            image = Image.open(i)
            image = transform(image).unsqueeze(0)
            image = image.to(DEVICE)
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()
            # Print the predicted class
            print("Predicted class:", predicted_class)
            # Append true and predicted labels to their respective lists
            true_classs.append(true_class)
            predicted_labels.append(predicted_class)
            
            # Check if the prediction is correct
            if predicted_class == true_class:
                correct_predictions += 1

    # Calculate accuracy and f1 socre
    accuracy = correct_predictions / total_predictions
    print("Accuracy:", accuracy)
    f1 = f1_score(true_classs, predicted_labels, average='weighted')
    print("Weighted F1 Score:", f1)

# Call predict_image function
predict_image(image_path, model, preprocess)

# Convert the lists to tensors
predicted_labels_tensor = torch.tensor(predicted_labels)
true_classs_tensor = torch.tensor(true_classs)

conf_matrix = ConfusionMatrix(num_classes=NUM_CLASSES, task='multiclass')
conf_matrix.update(predicted_labels_tensor, true_classs_tensor)

# Plot confusion matrix
conf_matrix.plot()
plt.show()
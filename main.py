import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.models as models

# Define hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 20

# Specify the path to your 'data' folder
data_dir = '.\data'

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the data
data_transforms = data_transforms['train']
image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'Cerebral Palsy'), data_transforms)
dataloaders = DataLoader(image_datasets, batch_size=batch_size, shuffle=True)

# Define the model
model = models.resnet18(pretrained=True)  # You can choose a different pre-trained model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(image_datasets.classes))  # Adjust output layer for your number of classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloaders)}')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')

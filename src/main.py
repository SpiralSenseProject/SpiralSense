import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
from models import *

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
STEP_SIZE = 10
GAMMA = 0.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_PRINT = 100

# Load and preprocess the data
data_dir = r"17flowers/jpg"

# Define transformation for preprocessing
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

augmentation = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomRotation(degrees=45),  # Random rotation
    transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
    transforms.RandomGrayscale(p=0.1),  # Random grayscale
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Random color jitter
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load the dataset using ImageFolder
original_dataset = ImageFolder(root=data_dir, transform=preprocess)
augmented_dataset = ImageFolder(root=data_dir, transform=augmentation)
dataset = original_dataset + augmented_dataset

print(dataset.datasets)
print("Length of dataset: ", len(dataset))

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the custom dataset
train_loader = DataLoader(CustomDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(CustomDataset(val_dataset), batch_size=BATCH_SIZE, num_workers=0)


# Initialize model, criterion, optimizer, and scheduler
model = VGG16(num_classes=3)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.8, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
print(dataset.datasets)

# Training loop
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if (i + 1) % NUM_PRINT == 0:
            print('[Epoch %d, Batch %d] Loss: %.6f' % (epoch + 1, i + 1, running_loss / NUM_PRINT))
            running_loss = 0.0

    # Print average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    print('[Epoch %d] Average Loss: %.6f' % (epoch + 1, avg_loss))

    lr_1 = optimizer.param_groups[0]['lr']
    print("Learning Rate: {:.15f}".format(lr_1))
    scheduler.step()

# Validation loop
val_loss = 0.0
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for data, targets in valid_loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

# Average validation loss
val_loss /= len(valid_loader)
print('Average Validation Loss: {:.6f}'.format(val_loss))

# Save the model
model_save_path = 'model.pth'
torch.save(model.state_dict(), model_save_path)
print('Model saved at', model_save_path)
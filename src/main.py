import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from models import *

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
STEP_SIZE = 10
GAMMA = 0.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_PRINT = 100

# Load and preprocess the data
data_dir = r"test_data/Task 1"

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
model = resnet18(pretrained=False, num_classes=2)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Lists to store training and validation loss history
TRAIN_LOSS_HIST = []
VAL_LOSS_HIST = []

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train(True)  # Set model to training mode
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
    
    # Calculate the average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    TRAIN_LOSS_HIST.append(loss.item())
    
    # Print average training loss for the epoch
    print('[Epoch %d] Average Training Loss: %.6f' % (epoch + 1, avg_train_loss))
    
    # Learning rate scheduling
    lr_1 = optimizer.param_groups[0]['lr']
    print("Learning Rate: {:.15f}".format(lr_1))
    scheduler.step()

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    # Calculate the average validation loss for the epoch
    avg_val_loss = val_loss / len(valid_loader)
    VAL_LOSS_HIST.append(loss.item())

    # Print average validation loss for the epoch
    print('Average Validation Loss: %.6f' % (avg_val_loss))

# End of training loop

# Save the model
model_save_path = 'model.pth'
torch.save(model.state_dict(), model_save_path)
print('Model saved at', model_save_path)

# Generate and display the loss plot
print('Generating loss plot...')
plt.plot(range(1, NUM_EPOCHS + 1), TRAIN_LOSS_HIST, label='Train Loss')
plt.plot(range(1, NUM_EPOCHS + 1), VAL_LOSS_HIST, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

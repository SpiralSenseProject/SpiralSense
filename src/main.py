import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
# Constants
RANDOM_SEED = 123
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
STEP_SIZE = 10
GAMMA = 0.5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_PRINT = 100

# Load and preprocess the data
data_dir = r"data\train\Task 1"

# Define transformation for preprocessing
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load the dataset using ImageFolder
dataset = ImageFolder(root=data_dir, transform=preprocess)

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
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for the custom dataset
train_loader = DataLoader(CustomDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(CustomDataset(val_dataset), batch_size=BATCH_SIZE, num_workers=0)
    #VGG16 model
class VGG16(torch.nn.Module):

        def __init__(self, num_classes):
            super().__init__()
            
            self.block_1 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
            )
            
            self.block_2 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=128,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
            )
            
            self.block_3 = torch.nn.Sequential(        
                    torch.nn.Conv2d(in_channels=128,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),        
                    torch.nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
            )
            
            
            self.block_4 = torch.nn.Sequential(   
                    torch.nn.Conv2d(in_channels=256,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),        
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),        
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),            
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))
            )
            
            self.block_5 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),            
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),            
                    torch.nn.Conv2d(in_channels=512,
                                    out_channels=512,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=1),
                    torch.nn.ReLU(),    
                    torch.nn.MaxPool2d(kernel_size=(2, 2),
                                    stride=(2, 2))             
            )
                
            height, width = 3, 3 
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512*height*width, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(p=0.5),
                torch.nn.Linear(4096, num_classes),
            )
                
            for m in self.modules():
                if isinstance(m, torch.torch.nn.Conv2d) or isinstance(m, torch.torch.nn.Linear):
                    torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.detach().zero_()
                        
            self.avgpool = torch.nn.AdaptiveAvgPool2d((height, width))
            
            
        def forward(self, x):

            x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = self.block_4(x)
            x = self.block_5(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1) # flatten
            
            logits = self.classifier(x)
            #probas = F.softmax(logits, dim=1)

            return logits     

        
# Initialize model, criterion, optimizer, and scheduler
model = VGG16(num_classes=5) 
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.8, weight_decay=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

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
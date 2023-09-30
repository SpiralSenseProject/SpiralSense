import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from models import *

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.00017588413773574044
STEP_SIZE = 10
GAMMA = 0.3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PRINT = 100
TASK = 1
RAW_DATA_DIR = r"data/train/raw/Task "
AUG_DATA_DIR = r"data/train/augmented/Task "
EXTERNAL_DATA_DIR = r"data/train/external/Task "
TEMP_DATA_DIR = "data/temp/"
NUM_CLASSES = 7
EARLY_STOPPING_PATIENCE = 20
CLASSES = ['Alzheimer Disease', 'Cerebral Palsy', 'Dystonia', 'Essential Tremor', 'Healthy', 'Huntington Disease', 'Parkinson Disease']
MODEL_SAVE_PATH = "output/checkpoints/model.pth"
MODEL = squeezenet1_0(num_classes=NUM_CLASSES)

print(CLASSES)


preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),  # Convert to tensor
        # Normalize 3 channels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, label

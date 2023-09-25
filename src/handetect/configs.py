import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from handetect.models import *

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.02750299610194638
STEP_SIZE = 10
GAMMA = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PRINT = 100
TASK = 1
ORIG_DATA_DIR = r"data/train/raw/Task " + str(TASK)
AUG_DATA_DIR = r"data/train/augmented/Task " + str(TASK)
NUM_CLASSES = len(os.listdir(ORIG_DATA_DIR))
MODEL_SAVE_PATH = "output/checkpoints/model.pth"
MODEL = shufflenet_v2_x0_5(num_classes=NUM_CLASSES)

preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
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

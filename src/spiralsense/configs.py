import torch
from torchvision import transforms
from torch.utils.data import Dataset
from models import *

# Constants
RANDOM_SEED = 123
BATCH_SIZE = 8
NUM_EPOCHS = 150
WARMUP_EPOCHS = 5
LEARNING_RATE = 0.0001
STEP_SIZE = 10
GAMMA = 0.3
CUTMIX_ALPHA = 0.3
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
NUM_PRINT = 100
TASK = 1
WARMUP_EPOCHS = 5
RAW_DATA_DIR = r"data/train/raw/Task "
AUG_DATA_DIR = r"data/train/augmented/Task "
EXTERNAL_DATA_DIR = r"data/train/external/Task "
COMBINED_DATA_DIR = r"data/train/combined/Task "
TEST_DATA_DIR = r"data/test/Task "
TEMP_DATA_DIR = "data/temp/Task "
NUM_CLASSES = 7
LABEL_SMOOTHING_EPSILON = 0.1
EARLY_STOPPING_PATIENCE = 20
CLASSES = [
    "Alzheimer Disease",
    "Cerebral Palsy",
    "Dystonia",
    "Essential Tremor",
    "Healthy",
    "Huntington Disease",
    "Parkinson Disease",
]


MODEL = EfficientNetB3WithNorm(num_classes=NUM_CLASSES)
MODEL_SAVE_PATH = r"output/checkpoints/" + MODEL.__class__.__name__ + ".pth"
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert to tensor
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
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

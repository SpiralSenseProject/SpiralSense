import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from models import *
from torch.utils.tensorboard import SummaryWriter #print to tensorboard
from torchvision.utils import make_grid
import optuna

writer = SummaryWriter()
# Constants
RANDOM_SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_PRINT = 100
TASK = 1
ORIG_DATA_DIR = r"data/train/raw/Task " + str(TASK)
AUG_DATA_DIR = r"data/train/augmented/Task " + str(TASK)
NUM_CLASSES = len(os.listdir(ORIG_DATA_DIR))
VAL_RESIZE_SIZE = 232

def resize_for_validation(image):
    return transforms.Resize((VAL_RESIZE_SIZE, VAL_RESIZE_SIZE))(image)

# Define transformation for preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
    ]
)

# Error if the classes in the original dataset and augmented dataset are not the same
assert (
    os.listdir(ORIG_DATA_DIR) == os.listdir(AUG_DATA_DIR)
), "Classes in original dataset and augmented dataset are not the same"


# Load the dataset using ImageFolder
original_dataset = ImageFolder(root=ORIG_DATA_DIR, transform=preprocess)
augmented_dataset = ImageFolder(root=AUG_DATA_DIR, transform=preprocess)
dataset = original_dataset + augmented_dataset

print("Classes: ", original_dataset.classes)
print("Length of original dataset: ", len(original_dataset))
print("Length of augmented dataset: ", len(augmented_dataset))
print("Length of total dataset: ", len(dataset))
print("Classes: ", original_dataset.classes)


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
train_loader = DataLoader(
    CustomDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
valid_loader = DataLoader(
    CustomDataset(val_dataset), batch_size=BATCH_SIZE, num_workers=0
)

# Initialize model, criterion, optimizer, and scheduler
model = mobilenet_v2(pretrained=False, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ReduceLROnPlateau scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10, verbose=True
)

# Lists to store training and validation loss history
TRAIN_LOSS_HIST = []
VAL_LOSS_HIST = []
AVG_TRAIN_LOSS_HIST = []
AVG_VAL_LOSS_HIST = []
TRAIN_ACC_HIST = []
VAL_ACC_HIST = []

def resize_for_validation(image):
    return transforms.Resize((VAL_RESIZE_SIZE, VAL_RESIZE_SIZE))(image)
    

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Modify the model and optimizer using suggested hyperparameters
    model = mobilenet_v2(pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(NUM_EPOCHS):
        model.train(True)  
    running_loss = 0.0
    total_train = 0
    correct_train = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % NUM_PRINT == 0:
            print(
                "[Epoch %d, Batch %d] Loss: %.6f"
                % (epoch + 1, i + 1, running_loss / NUM_PRINT)
            )
            running_loss = 0.0

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    TRAIN_LOSS_HIST.append(loss.item())
    train_accuracy = correct_train / total_train
    TRAIN_ACC_HIST.append(train_accuracy)
    # Calculate the average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    AVG_TRAIN_LOSS_HIST.append(avg_train_loss)

    # Print average training loss for the epoch
    print("[Epoch %d] Average Training Loss: %.6f" % (epoch + 1, avg_train_loss))

    # Learning rate scheduling
    lr_1 = optimizer.param_groups[0]["lr"]
    print("Learning Rate: {:.15f}".format(lr_1))
    scheduler.step(avg_train_loss)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    VAL_LOSS_HIST.append(loss.item())

    # Calculate the average validation loss for the epoch
    avg_val_loss = val_loss / len(valid_loader)
    AVG_VAL_LOSS_HIST.append(loss.item())
    print("Average Validation Loss: %.6f" % (avg_val_loss))

    # Calculate the accuracy of validation set
    val_accuracy = correct_val / total_val
    VAL_ACC_HIST.append(val_accuracy)
    print("Validation Accuracy: %.6f" % (val_accuracy))
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
    # Add sample images to TensorBoard
    sample_images, _ = next(iter(valid_loader))  # Get a batch of sample images
    sample_images = sample_images.to(DEVICE)
    grid_image = make_grid(sample_images, nrow=8, normalize=True)  # Create a grid of images
    writer.add_image('Sample Images', grid_image, global_step=epoch)
    # Validation loop
    model.eval()  # Set model to evaluation mode
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # suan evaluation score 
    evaluation_score = correct_val / total_val

    # Return the evaluation score 
    return evaluation_score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300, timeout=800)

    # Print statistics
    print("Number of finished trials: ", len(study.trials))
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print("Number of pruned trials: ", len(pruned_trials))
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Number of complete trials: ", len(complete_trials))

    # Print best trial
    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from models import *
from torch.utils.tensorboard import SummaryWriter
from configs import *
import data_loader

# Set up TensorBoard writer
writer = SummaryWriter(log_dir="output/tensorboard/training")

# Define a function for plotting and logging metrics
def plot_and_log_metrics(metrics_dict, step, prefix="Train"):
    for metric_name, metric_value in metrics_dict.items():
        writer.add_scalar(f"{prefix}/{metric_name}", metric_value, step)

# Data loader
train_loader, valid_loader = data_loader.load_data(
    RAW_DATA_DIR, AUG_DATA_DIR, EXTERNAL_DATA_DIR, preprocess
)

# Initialize model, criterion, optimizer, and scheduler
MODEL = MODEL.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# Lists to store training and validation loss history
TRAIN_LOSS_HIST = []
VAL_LOSS_HIST = []
AVG_TRAIN_LOSS_HIST = []
AVG_VAL_LOSS_HIST = []
TRAIN_ACC_HIST = []
VAL_ACC_HIST = []

# Training loop
for epoch in range(NUM_EPOCHS):
    MODEL.train()  # Set model to training mode
    running_loss = 0.0
    total_train = 0
    correct_train = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = MODEL(inputs)
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

    avg_train_loss = running_loss / len(train_loader)
    TRAIN_LOSS_HIST.append(avg_train_loss)
    TRAIN_ACC_HIST.append(correct_train / total_train)

    # Log training metrics
    train_metrics = {
        "Loss": avg_train_loss,
        "Accuracy": correct_train / total_train,
    }
    plot_and_log_metrics(train_metrics, epoch, prefix="Train")

    # Learning rate scheduling
    scheduler.step()

    # Validation loop
    MODEL.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = MODEL(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(valid_loader)
    VAL_LOSS_HIST.append(avg_val_loss)
    VAL_ACC_HIST.append(correct_val / total_val)

    # Log validation metrics
    val_metrics = {
        "Loss": avg_val_loss,
        "Accuracy": correct_val / total_val,
    }
    plot_and_log_metrics(val_metrics, epoch, prefix="Validation")

    # Add sample images to TensorBoard
    sample_images, _ = next(iter(valid_loader))
    sample_images = sample_images.to(DEVICE)
    grid_image = make_grid(
        sample_images, nrow=8, normalize=True
    )
    writer.add_image("Sample Images", grid_image, global_step=epoch)

# Save the model
torch.save(MODEL.state_dict(), MODEL_SAVE_PATH)
print("Model saved at", MODEL_SAVE_PATH)

# Plot loss and accuracy curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), TRAIN_LOSS_HIST, label="Train Loss")
plt.plot(range(1, NUM_EPOCHS + 1), VAL_LOSS_HIST, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curves")

plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), TRAIN_ACC_HIST, label="Train Accuracy")
plt.plot(range(1, NUM_EPOCHS + 1), VAL_ACC_HIST, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curves")

plt.tight_layout()
plt.savefig("training_curves.png")

# Close TensorBoard writer
writer.close()

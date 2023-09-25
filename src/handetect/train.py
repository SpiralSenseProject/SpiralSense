import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from models import *
from scipy.ndimage import gaussian_filter1d
from torch.utils.tensorboard import SummaryWriter  # print to tensorboard
from torchvision.utils import make_grid
import pandas as pd
from configs import *
import data_loader

# torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

writer = SummaryWriter(log_dir="output/tensorboard")


# Data loader
train_loader, valid_loader = data_loader.load_data(
    ORIG_DATA_DIR, AUG_DATA_DIR, preprocess
)


# Initialize model, criterion, optimizer, and scheduler
MODEL = MODEL.to(DEVICE)
criterion = nn.CrossEntropyLoss()
# Adam optimizer
optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
# StepLR scheduler
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
    MODEL.train(True)  # Set model to training mode
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

    TRAIN_ACC_HIST.append(correct_train / total_train)

    TRAIN_LOSS_HIST.append(loss.item())

    # Calculate the average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/Train", correct_train / total_train, epoch)
    AVG_TRAIN_LOSS_HIST.append(avg_train_loss)

    # Print average training loss for the epoch
    print("[Epoch %d] Average Training Loss: %.6f" % (epoch + 1, avg_train_loss))

    # Learning rate scheduling
    lr_1 = optimizer.param_groups[0]["lr"]
    print("Learning Rate: {:.15f}".format(lr_1))
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

    VAL_LOSS_HIST.append(loss.item())

    # Calculate the average validation loss for the epoch
    avg_val_loss = val_loss / len(valid_loader)
    AVG_VAL_LOSS_HIST.append(loss.item())
    print("Average Validation Loss: %.6f" % (avg_val_loss))

    # Calculate the accuracy of validation set
    val_accuracy = correct_val / total_val
    VAL_ACC_HIST.append(val_accuracy)
    print("Validation Accuracy: %.6f" % (val_accuracy))

    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
    # Add sample images to TensorBoard
    sample_images, _ = next(iter(valid_loader))  # Get a batch of sample images
    sample_images = sample_images.to(DEVICE)
    grid_image = make_grid(
        sample_images, nrow=8, normalize=True
    )  # Create a grid of images
    writer.add_image("Sample Images", grid_image, global_step=epoch)

# End of training loop

# Save the model

torch.save(MODEL.state_dict(), MODEL_SAVE_PATH)
print("Model saved at", MODEL_SAVE_PATH)

print("Generating loss plot...")
# train_loss_line = gaussian_filter1d(TRAIN_LOSS_HIST, sigma=10)
# val_loss_line = gaussian_filter1d(VAL_LOSS_HIST, sigma=10)
# plt.plot(range(1, NUM_EPOCHS + 1), train_loss_line, label='Train Loss')
# plt.plot(range(1, NUM_EPOCHS + 1), val_loss_line, label='Validation Loss')
avg_train_loss_line = gaussian_filter1d(AVG_TRAIN_LOSS_HIST, sigma=2)
avg_val_loss_line = gaussian_filter1d(AVG_VAL_LOSS_HIST, sigma=2)
train_loss_line = gaussian_filter1d(TRAIN_LOSS_HIST, sigma=2)
val_loss_line = gaussian_filter1d(VAL_LOSS_HIST, sigma=2)
train_acc_line = gaussian_filter1d(TRAIN_ACC_HIST, sigma=2)
val_acc_line = gaussian_filter1d(VAL_ACC_HIST, sigma=2)
plt.plot(range(1, NUM_EPOCHS + 1), train_loss_line, label="Train Loss")
plt.plot(range(1, NUM_EPOCHS + 1), val_loss_line, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train Loss and Validation Loss")
plt.savefig("loss_plot.png")
plt.clf()
plt.plot(range(1, NUM_EPOCHS + 1), avg_train_loss_line, label="Average Train Loss")
plt.plot(range(1, NUM_EPOCHS + 1), avg_val_loss_line, label="Average Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Average Train Loss and Average Validation Loss")
plt.savefig("avg_loss_plot.png")
plt.clf()
plt.plot(range(1, NUM_EPOCHS + 1), train_acc_line, label="Train Accuracy")
plt.plot(range(1, NUM_EPOCHS + 1), val_acc_line, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Train Accuracy and Validation Accuracy")
plt.savefig("accuracy_plot.png")

dummy_input = torch.randn(1, 3, 64, 64).to(DEVICE)  # Adjust input shape accordingly
writer.add_graph(MODEL, dummy_input)
# Close TensorBoard writer
writer.close()

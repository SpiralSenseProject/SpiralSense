import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import *  # Import your model here
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import optuna
from configs import *
import data_loader

# Data loader
train_loader, valid_loader = data_loader.load_data(
    RAW_DATA_DIR, AUG_DATA_DIR, EXTERNAL_DATA_DIR, preprocess
)

# Initialize model, criterion, optimizer, and scheduler
MODEL = MODEL.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10, verbose=True
)

# Lists to store training and validation loss history
TRAIN_LOSS_HIST = []
VAL_LOSS_HIST = []
TRAIN_ACC_HIST = []
VAL_ACC_HIST = []
AVG_TRAIN_LOSS_HIST = []
AVG_VAL_LOSS_HIST = []

# Create a TensorBoard writer for logging
writer = SummaryWriter(
    log_dir="output/tensorboard/tuning",
)

# Define early stopping parameters
early_stopping_patience = 10  # Number of epochs to wait for improvement
best_val_loss = float('inf')
no_improvement_count = 0

def train_epoch(epoch):
    MODEL.train(True)
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

    TRAIN_LOSS_HIST.append(loss.item())
    train_accuracy = correct_train / total_train
    TRAIN_ACC_HIST.append(train_accuracy)
    # Calculate the average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)

    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
    AVG_TRAIN_LOSS_HIST.append(avg_train_loss)

    # Print average training loss for the epoch
    print("[Epoch %d] Average Training Loss: %.6f" % (epoch + 1, avg_train_loss))

    # Learning rate scheduling
    lr_1 = optimizer.param_groups[0]["lr"]
    print("Learning Rate: {:.15f}".format(lr_1))
    scheduler.step(avg_train_loss)

def validate_epoch(epoch):
    global best_val_loss, no_improvement_count  
    
    MODEL.eval()
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

    # Calculate the accuracy of the validation set
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

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    if no_improvement_count >= early_stopping_patience:
        print(f"Early stopping after {epoch + 1} epochs without improvement.")
        return True  # Return True to stop training

def objective(trial):
    global best_val_loss, no_improvement_count
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Modify the model and optimizer using suggested hyperparameters
    optimizer = optim.Adam(MODEL.parameters(), lr=learning_rate)

    for epoch in range(20):
        train_epoch(epoch)
        early_stopping = validate_epoch(epoch)

        # Check for early stopping
        if early_stopping:
            break

    # Calculate a weighted score based on validation accuracy and loss
    validation_score = VAL_ACC_HIST[-1] - AVG_VAL_LOSS_HIST[-1]

    # Return the negative score as Optuna maximizes by default
    return -validation_score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=3600)

    # Print statistics
    print("Number of finished trials: ", len(study.trials))
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    print("Number of pruned trials: ", len(pruned_trials))
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    print("Number of complete trials: ", len(complete_trials))

    # Print best trial
    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", -trial.value)  # Negate the value as it was maximized
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Close TensorBoard writer
    writer.close()

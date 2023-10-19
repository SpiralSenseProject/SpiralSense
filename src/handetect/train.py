import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import rcParams
from models import *
from torch.utils.tensorboard import SummaryWriter
from configs import *
import data_loader
import torch.nn.functional as F
import csv
import numpy as np
from torchcontrib.optim import SWA


rcParams["font.family"] = "Times New Roman"

SWA_START = 5  # Starting epoch for SWA
SWA_FREQ = 5  # Frequency of updating SWA weights


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(input, target, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = input.size()[0]
    index = torch.randperm(batch_size)
    rand_index = torch.randperm(input.size()[0])

    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    targets_a = target
    targets_b = target[rand_index]

    return input, targets_a, targets_b, lam


def cutmix_criterion(criterion, outputs, targets_a, targets_b, lam):
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
        outputs, targets_b
    )


def setup_tensorboard():
    return SummaryWriter(log_dir="output/tensorboard/training")


def load_and_preprocess_data():
    return data_loader.load_data(
        COMBINED_DATA_DIR + "1",
        preprocess,
    )


def initialize_model_optimizer_scheduler():
    model = MODEL.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return model, criterion, optimizer, scheduler


def plot_and_log_metrics(metrics_dict, step, writer, prefix="Train"):
    for metric_name, metric_value in metrics_dict.items():
        writer.add_scalar(f"{prefix}/{metric_name}", metric_value, step)


def train_one_epoch(model, criterion, optimizer, train_loader, epoch, alpha):
    model.train()
    running_loss = 0.0
    total_train = 0
    correct_train = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # Apply CutMix
        inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1)

        outputs = model(inputs)

        # Calculate CutMix loss
        loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if (i + 1) % NUM_PRINT == 0:
            print(
                f"[Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}] "
                f"Loss: {running_loss / NUM_PRINT:.6f}"
            )
            running_loss = 0.0

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss, correct_train / total_train


def validate_model(model, criterion, valid_loader):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(valid_loader)
    return avg_val_loss, correct_val / total_val


def main_training_loop():
    writer = setup_tensorboard()
    train_loader, valid_loader = load_and_preprocess_data()
    model, criterion, optimizer, scheduler = initialize_model_optimizer_scheduler()

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    no_improvement_count = 0
    epoch_metrics = []

    AVG_TRAIN_LOSS_HIST = []
    AVG_VAL_LOSS_HIST = []
    TRAIN_ACC_HIST = []
    VAL_ACC_HIST = []

    # Initialize SWA optimizer
    swa_optimizer = SWA(optimizer, swa_start=SWA_START, swa_freq=SWA_FREQ)

    for epoch in range(NUM_EPOCHS):
        print(f"\n[Epoch: {epoch + 1}/{NUM_EPOCHS}]")
        print("Learning rate:", scheduler.get_last_lr()[0])

        avg_train_loss, train_accuracy = train_one_epoch(
            model, criterion, optimizer, train_loader, epoch, CUTMIX_ALPHA
        )
        AVG_TRAIN_LOSS_HIST.append(avg_train_loss)
        TRAIN_ACC_HIST.append(train_accuracy)

        # Log training metrics
        train_metrics = {
            "Loss": avg_train_loss,
            "Accuracy": train_accuracy,
        }
        plot_and_log_metrics(train_metrics, epoch, writer=writer, prefix="Train")
        epoch_metrics.append(
            {
                "Epoch": epoch + 1,
                "Train Loss": avg_train_loss,
                "Train Accuracy": train_accuracy,
                "Validation Loss": avg_val_loss,
                "Validation Accuracy": val_accuracy,
                "Learning Rate": scheduler.get_last_lr()[0],
            }
        )

        # Learning rate scheduling

        if epoch < WARMUP_EPOCHS:
            # Linear warm-up phase
            lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # Cosine annealing scheduler after warm-up
            scheduler.step()

        avg_val_loss, val_accuracy = validate_model(model, criterion, valid_loader)
        AVG_VAL_LOSS_HIST.append(avg_val_loss)
        VAL_ACC_HIST.append(val_accuracy)

        # Log validation metrics
        val_metrics = {
            "Loss": avg_val_loss,
            "Accuracy": val_accuracy,
        }
        plot_and_log_metrics(val_metrics, epoch, writer=writer, prefix="Validation")

        # Print average training and validation metrics
        print(f"Average Training Loss: {avg_train_loss:.6f}")
        print(f"Average Validation Loss: {avg_val_loss:.6f}")
        print(f"Training Accuracy: {train_accuracy:.6f}")
        print(f"Validation Accuracy: {val_accuracy:.6f}")

        # Check for early stopping based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Early stopping condition
        if no_improvement_count >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: Validation accuracy did not improve for {} consecutive epochs.".format(
                    EARLY_STOPPING_PATIENCE
                )
            )
            break

        # Update SWA weights
        if epoch >= SWA_START and epoch % SWA_FREQ == 0:
            swa_optimizer.update_swa()

    # Apply SWA to the final model weights
    swa_optimizer.swap_swa_sgd()
    csv_filename = "training_metrics.csv"

    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = [
            "Epoch",
            "Train Loss",
            "Train Accuracy",
            "Validation Loss",
            "Validation Accuracy",
            "Learning Rate",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for metric in epoch_metrics:
            writer.writerow(metric)

    print(f"Metrics saved to {csv_filename}")

    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("\nModel saved at", MODEL_SAVE_PATH)

    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, len(AVG_TRAIN_LOSS_HIST) + 1),
        AVG_TRAIN_LOSS_HIST,
        label="Average Train Loss",
    )
    plt.plot(
        range(1, len(AVG_VAL_LOSS_HIST) + 1),
        AVG_VAL_LOSS_HIST,
        label="Average Validation Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(TRAIN_ACC_HIST) + 1), TRAIN_ACC_HIST, label="Train Accuracy")
    plt.plot(range(1, len(VAL_ACC_HIST) + 1), VAL_ACC_HIST, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curves")

    plt.tight_layout()
    plt.savefig("training_curves.png")

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main_training_loop()

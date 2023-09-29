import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import *
from torch.utils.tensorboard import SummaryWriter
from configs import *
import data_loader


def setup_tensorboard():
    return SummaryWriter(log_dir="output/tensorboard/training")


def load_and_preprocess_data():
    return data_loader.load_data(
        RAW_DATA_DIR + str(TASK), AUG_DATA_DIR + str(TASK), EXTERNAL_DATA_DIR + str(TASK), preprocess
    )


def initialize_model_optimizer_scheduler():
    model = MODEL.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    return model, criterion, optimizer, scheduler


def plot_and_log_metrics(metrics_dict, step, writer, prefix="Train"):
    for metric_name, metric_value in metrics_dict.items():
        writer.add_scalar(f"{prefix}/{metric_name}", metric_value, step)


def train_one_epoch(model, criterion, optimizer, train_loader, epoch):
    model.train()
    running_loss = 0.0
    total_train = 0
    correct_train = 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if model.__class__.__name__ == "GoogLeNet":
            outputs = model(inputs).logits
        else:
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

    AVG_TRAIN_LOSS_HIST = []
    AVG_VAL_LOSS_HIST = []
    TRAIN_ACC_HIST = []
    VAL_ACC_HIST = []

    for epoch in range(NUM_EPOCHS):
        print(f"[Epoch: {epoch + 1}]")
        print("Learning rate:", scheduler.get_last_lr()[0])

        avg_train_loss, train_accuracy = train_one_epoch(
            model, criterion, optimizer, train_loader, epoch
        )
        AVG_TRAIN_LOSS_HIST.append(avg_train_loss)
        TRAIN_ACC_HIST.append(train_accuracy)

        # Log training metrics
        train_metrics = {
            "Loss": avg_train_loss,
            "Accuracy": train_accuracy,
        }
        plot_and_log_metrics(train_metrics, epoch, writer=writer, prefix="Train")

        # Learning rate scheduling
        scheduler.step()

        avg_val_loss, val_accuracy = validate_model(model, criterion, valid_loader)
        AVG_VAL_LOSS_HIST.append(avg_val_loss)
        VAL_ACC_HIST.append(val_accuracy)

        # Log validation metrics
        val_metrics = {
            "Loss": avg_val_loss,
            "Accuracy": val_accuracy,
        }
        plot_and_log_metrics(train_metrics, epoch, writer=writer, prefix="Train")

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

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved at", MODEL_SAVE_PATH)

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

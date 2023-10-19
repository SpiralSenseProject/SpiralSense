import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from configs import *
import data_loader
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

torch.cuda.empty_cache()

print(f"Using device: {DEVICE}")

EPOCHS = 10
# N_TRIALS = 10
# TIMEOUT = 5000

EARLY_STOPPING_PATIENCE = (
    4  # Number of epochs with no improvement to trigger early stopping
)


# Create a TensorBoard writer
writer = SummaryWriter(log_dir="output/tensorboard/tuning")


# Function to create or modify data loaders with the specified batch size
def create_data_loaders(batch_size):
    train_loader, valid_loader = data_loader.load_data(
        COMBINED_DATA_DIR + "1",
        preprocess,
        batch_size=batch_size,
    )
    return train_loader, valid_loader


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


# Objective function for optimization
def objective(trial, model=MODEL):
    model = model.to(DEVICE)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    train_loader, valid_loader = create_data_loaders(batch_size)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    gamma = trial.suggest_float("gamma", 0.1, 0.9, step=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    past_trials = 0  # Number of trials already completed

    # Print best hyperparameters:
    if past_trials > 0:
        print("\nBest Hyperparameters:")
        print(f"{study.best_trial.params}")

    print(f"\n[INFO] Trial: {trial.number}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Gamma: {gamma}\n")

    early_stopping_counter = 0
    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            if model.__class__.__name__ == "GoogLeNet":
                output = model(data).logits
            else:
                output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader, 0):
                data, target = data.to(DEVICE), target.to(DEVICE)
                data, targets_a, targets_b, lam = cutmix_data(data, target, alpha=1)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(valid_loader.dataset)
        if accuracy >= 1.0:
            print(f"Desired accuracy of 1.0 achieved. Stopping early.")
            return float("inf")

        # Log hyperparameters and accuracy to TensorBoard
        writer.add_scalar("Accuracy", accuracy, trial.number)
        writer.add_hparams(
            {"batch_size": batch_size, "lr": lr, "gamma": gamma},
            {"accuracy": accuracy},
        )

        print(f"[EPOCH {epoch + 1}] Accuracy: {accuracy:.4f}")

        trial.report(accuracy, epoch)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Early stopping check
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    if trial.number > 10 and trial.params["lr"] < 1e-3 and best_accuracy < 0.7:
        return float("inf")

    past_trials += 1

    return best_accuracy


if __name__ == "__main__":
    hyperband_pruner = optuna.pruners.HyperbandPruner()

    # Record the start time
    start_time = time.time()

    # storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(
        direction="maximize",
        pruner=hyperband_pruner,
        study_name="hyperparameter_tuning",
        storage="sqlite:///" + MODEL.__class__.__name__ + ".sqlite3",
    )

    study.optimize(objective)

    # Record the end time
    end_time = time.time()

    # Calculate the duration of hyperparameter tuning
    tuning_duration = end_time - start_time
    print(f"Hyperparameter tuning took {tuning_duration:.2f} seconds.")

    best_trial = study.best_trial
    print("\nBest Trial:")
    print(f"  Trial Number: {best_trial.number}")
    print(f"  Best Accuracy: {best_trial.value:.4f}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

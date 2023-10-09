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

torch.cuda.empty_cache()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
N_TRIALS = 21600
TIMEOUT = 21600
EARLY_STOPPING_PATIENCE = (
    4  # Number of epochs with no improvement to trigger early stopping
)


# Create a TensorBoard writer
writer = SummaryWriter(log_dir="output/tensorboard/tuning")


# Function to create or modify data loaders with the specified batch size
def create_data_loaders(batch_size):
    train_loader, valid_loader = data_loader.load_data(
        RAW_DATA_DIR + str(TASK),
        AUG_DATA_DIR + str(TASK),
        EXTERNAL_DATA_DIR + str(TASK),
        preprocess,
        batch_size=batch_size,
    )
    return train_loader, valid_loader


# Objective function for optimization
def objective(trial, model=model):
    model = model.to(DEVICE)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader, valid_loader = create_data_loaders(batch_size)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    gamma = trial.suggest_float("gamma", 0.1, 0.9, step=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

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
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(valid_loader.dataset)

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

    return best_accuracy


if __name__ == "__main__":
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="hyperparameter_tuning",
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    best_trial = study.best_trial
    print("\nBest Trial:")
    print(f"  Trial Number: {best_trial.number}")
    print(f"  Best Accuracy: {best_trial.value:.4f}")
    print("  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

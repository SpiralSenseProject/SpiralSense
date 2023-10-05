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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
N_TRIALS = 1000
TIMEOUT = 1800

# Create a TensorBoard writer
writer = SummaryWriter(log_dir="output/tensorboard/tuning")


def create_data_loaders(batch_size):
    # Create or modify data loaders with the specified batch size
    train_loader, valid_loader = data_loader.load_data(
        RAW_DATA_DIR + str(TASK),
        AUG_DATA_DIR + str(TASK),
        EXTERNAL_DATA_DIR + str(TASK),
        preprocess,
        batch_size=batch_size,
    )
    return train_loader, valid_loader


def objective(trial, model=MODEL):
    # Generate the model.
    model = model.to(DEVICE)

    # Suggest batch size for tuning.
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Create data loaders with the suggested batch size.
    train_loader, valid_loader = create_data_loaders(batch_size)

    # Generate the optimizer.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Suggest the gamma parameter for the learning rate scheduler.
    gamma = trial.suggest_float("gamma", 0.1, 1.0, step=0.1)

    # Create a learning rate scheduler with the suggested gamma.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    # Training of the model.
    for epoch in range(EPOCHS):
        print(f"[Epoch: {epoch} | Trial: {trial.number}]")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            if (
                model.__class__.__name__ == "GoogLeNet"
            ):  # the shit GoogLeNet has a different output
                output = model(data).logits
            else:
                output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Update the learning rate using the scheduler.
        scheduler.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader, 0):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(valid_loader.dataset)

        # Log hyperparameters and accuracy to TensorBoard
        writer.add_scalar("Accuracy", accuracy, trial.number)
        writer.add_hparams(
            {"batch_size": batch_size, "lr": lr, "gamma": gamma},
            {"accuracy": accuracy},
        )

        # Print hyperparameters and accuracy
        print("Hyperparameters: ", trial.params)
        print("Accuracy: ", accuracy)
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if trial.number > 10 and trial.params["lr"] < 1e-3 and accuracy < 0.7:
        return float("inf")  # Prune the trial

    return accuracy


if __name__ == "__main__":
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="maximize",  # Adjust the direction as per your optimization goal
        pruner=pruner,
        study_name="hyperparameter_tuning",
    )

    # Optimize the hyperparameters
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT)

    # Print the best trial
    best_trial = study.best_trial
    print("Best trial:")
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

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

optuna.logging.set_verbosity(optuna.logging.DEBUG)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10

# Create a TensorBoard writer
writer = SummaryWriter(log_dir="output/tensorboard/tuning/", )

def create_data_loaders(batch_size):
    # Create or modify data loaders with the specified batch size
    train_loader, valid_loader = data_loader.load_data(
        RAW_DATA_DIR, AUG_DATA_DIR, EXTERNAL_DATA_DIR, preprocess, batch_size=batch_size
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
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training of the model.
    for epoch in range(EPOCHS):
        print(f"[Epoch: {epoch} | Trial: {trial.number}]")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            if model.__class__.__name__ == "GoogLeNet": # the shit GoogLeNet has a different output
                output = model(data).logits
            else:
                output = model(data)
            loss = criterion(output, target)
            loss.backward()
            if optimizer_name == "LBFGS":
                optimizer.step(closure=lambda: loss)
            else:
                optimizer.step()

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
            {
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "lr": lr
            },
            {
                "accuracy": accuracy
            }
        )

        # Print hyperparameters and accuracy
        print("Hyperparameters: ", trial.params)
        print("Accuracy: ", accuracy)
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="handetect")
    study.optimize(objective, n_trials=100, timeout=1000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Close TensorBoard writer
    writer.close()

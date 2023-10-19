import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
from configs import *
import data_loader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pygad
import pygad.torchga

torch.cuda.empty_cache()
model = MODEL.to(DEVICE)

EPOCHS = 10
N_TRIALS = 20
TIMEOUT = 1800
EARLY_STOPPING_PATIENCE = (
    4  # Number of epochs with no improvement to trigger early stopping
)
NUM_GENERATIONS = 10
SOL_PER_POP = 10  # Number of solutions in the population
NUM_GENES = 2
NUM_PARENTS_MATING = 4

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


# Objective function for optimization
def objective(trial):
    global data_inputs, data_outputs

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
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

    past_trials += 1

    return best_accuracy


# Custom genetic algorithm
def run_genetic_algorithm(fitness_func):
    # Initial population
    population = np.random.rand(SOL_PER_POP, NUM_GENES)  # Random initialization

    # Run for a fixed number of generations
    for generation in range(NUM_GENERATIONS):
        # Calculate fitness for each solution in the population
        fitness = np.array(
            [fitness_func(solution, idx) for idx, solution in enumerate(population)]
        )

        # Get the index of the best solution
        best_idx = np.argmax(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        # Print the best solution and fitness for this generation
        print(f"Generation {generation + 1}:")
        print("Best Solution:")
        print("Learning Rate = {lr}".format(lr=best_solution[0]))
        print("Gamma = {gamma}".format(gamma=best_solution[1]))
        print("Best Fitness = {fitness}".format(fitness=best_fitness))

        # Perform selection and crossover to create the next generation
        population = selection_and_crossover(population, fitness)


# Selection and crossover logic
def selection_and_crossover(population, fitness):
    # Perform tournament selection
    parents = []
    for _ in range(SOL_PER_POP):
        tournament_idxs = np.random.choice(range(SOL_PER_POP), NUM_PARENTS_MATING)
        tournament_fitness = [fitness[idx] for idx in tournament_idxs]
        selected_parent_idx = tournament_idxs[np.argmax(tournament_fitness)]
        parents.append(population[selected_parent_idx])

    # Perform single-point crossover
    offspring = []
    for i in range(0, SOL_PER_POP, 2):
        if i + 1 < SOL_PER_POP:
            crossover_point = np.random.randint(0, NUM_GENES)
            offspring.extend(
                [
                    np.concatenate(
                        (parents[i][:crossover_point], parents[i + 1][crossover_point:])
                    )
                ]
            )
            offspring.extend(
                [
                    np.concatenate(
                        (parents[i + 1][:crossover_point], parents[i][crossover_point:])
                    )
                ]
            )

    return np.array(offspring)


# Modify callback function to log best accuracy
def callback_generation(ga_instance):
    global study

    # Fetch the parameters of the best solution
    solution, solution_fitness, _ = ga_instance.best_solution()
    best_learning_rate, best_gamma = solution

    # Report the best accuracy to Optuna study
    study.set_user_attr("best_accuracy", solution_fitness)

    # Print generation number and best fitness
    print(
        "Generation = {generation}".format(generation=ga_instance.generations_completed)
    )
    print("Best Fitness = {fitness}".format(fitness=solution_fitness))
    print("Best Learning Rate = {lr}".format(lr=best_learning_rate))
    print("Best Gamma = {gamma}".format(gamma=best_gamma))


if __name__ == "__main__":
    global study
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="hyperparameter_tuning",
    )

    # Define data_inputs and data_outputs
    # You need to populate these with your own data

    # Define the loss function
    loss_function = nn.CrossEntropyLoss()

    def fitness_func(solution, sol_idx):
        global data_inputs, data_outputs, model, loss_function

        learning_rate, momentum = solution

        # Update optimizer with the current learning rate and momentum
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=momentum
        )

        # Load the model weights
        model_weights_dict = pygad.torchga.model_weights_as_dict(
            model=model, weights_vector=solution
        )
        model.load_state_dict(model_weights_dict)

        # Forward pass
        predictions = model(data_inputs)

        # Calculate cross-entropy loss
        loss = loss_function(predictions, data_outputs)

        # Higher fitness for lower loss
        solution_fitness = 1.0 / (loss.detach().numpy() + 1e-8)

        return solution_fitness

    # Run the custom genetic algorithm
    run_genetic_algorithm(fitness_func)

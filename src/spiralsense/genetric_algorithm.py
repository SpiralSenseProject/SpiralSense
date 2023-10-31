import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from configs import *
import data_loader
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random

torch.cuda.empty_cache()
RANDOM_SEED1=42
random.seed(RANDOM_SEED1)
torch.cuda.manual_seed(RANDOM_SEED1)
torch.manual_seed(RANDOM_SEED1)
print("PyTorch Seed:", torch.initial_seed())
print("Random Seed:", random.getstate()[1][0])
print("PyTorch CUDA Seed:", torch.cuda.initial_seed())


# Define the constants for genetic algorithm
POPULATION_SIZE = 5
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
NUM_GENERATIONS = 5

EPOCHS = 5

EARLY_STOPPING_PATIENCE = 4
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

# Create a TensorBoard writer
writer = SummaryWriter(log_dir="output/tensorboard/tuning")
model = MODEL.to(DEVICE)
# model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

def fitness_function(individual,model):
    batch_size, lr,= individual
    
    # Assuming you have a model, optimizer, and loss function defined
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    # Define your data loaders using the given batch_size
    train_loader, valid_loader = create_data_loaders(batch_size)

    # Training loop
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
        # Validation loop
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
        print(f"Epoch {epoch + 1}/{EPOCHS}, Accuracy: {accuracy:.4f}")
    return accuracy,

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
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

# Function to create or modify data loaders with the specified batch size
def create_data_loaders(batch_size):
    print(f"Batch Size (before conversion): {batch_size}")
    batch_size = int(batch_size)  # Ensure batch_size is an integer
    print(f"Batch Size (after conversion): {batch_size}")
    train_loader, valid_loader = data_loader.load_data(
        COMBINED_DATA_DIR + "1",
        preprocess,
        batch_size=batch_size,
    )
    return train_loader, valid_loader

# Genetic algorithm initialization functions
def create_individual():
    lr = abs(np.random.uniform(0.0006, 0.0009))
    print(f"Generated lr: {lr}")
    return creator.Individual([
        int(np.random.choice([32])),  # Choose a valid batch size
        lr,  # lr in log scale between 1e-4 and 1e-2
    ])
# Genetic algorithm evaluation function
def evaluate_individual(individual, model=MODEL):
    batch_size, lr, = individual
    lr=abs(lr)
    # Assuming you have a model, optimizer, and loss function defined
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Define your data loaders using the given batch_size
    train_loader, valid_loader = create_data_loaders(batch_size)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            # Apply CutMix
            data, targets_a, targets_b, lam = cutmix_data(data, target, alpha=1)
            if model.__class__.__name__ == "GoogLeNet":
                output = model(data).logits
            else:
                output = model(data)
            loss = cutmix_criterion(criterion, output, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()

        # Validation loop
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

        # Log accuracy or other metrics as needed
        writer.add_scalar("Accuracy", accuracy, epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Accuracy: {accuracy:.4f}")

    # Return the accuracy (or any other metric you want to optimize)
    return (accuracy,)

if __name__ == "__main__":
    pruner = optuna.pruners.HyperbandPruner()

    start_time = time.time()
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name="hyperparameter_optimization",
        storage="sqlite:///" + MODEL.__class__.__name__ + ".sqlite3",
    )

    from deap import base, creator, tools, algorithms

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_function, model=model)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=MUTATION_RATE)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POPULATION_SIZE)

    for ind in population:
        print(type(ind))
        fitness_value = evaluate_individual(ind, model)
        ind.fitness.values = (fitness_value[0],)

    algorithms.eaSimple(population, toolbox, cxpb=CROSSOVER_RATE, mutpb=MUTATION_RATE, ngen=NUM_GENERATIONS, stats=None, halloffame=None, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    best_batch_size, best_lr = best_individual

    best_accuracy = evaluate_individual(best_individual, model)

    print("Best Hyperparameters:")
    print(f"Batch Size: {best_batch_size}")
    print(f"Learning Rate: {best_lr}")
    print(f"Best Accuracy: {best_accuracy[0]}")

    end_time = time.time()
    tuning_duration = end_time - start_time
    print(f"Hyperparameter optimization took {tuning_duration:.2f} seconds.")
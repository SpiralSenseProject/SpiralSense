from configs import *
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, Dataset
import torch

torch.manual_seed(RANDOM_SEED)


def load_data(combined_dir, preprocess, batch_size=BATCH_SIZE):
    dataset = ImageFolder(combined_dir, transform=preprocess)

    # Classes
    classes = dataset.classes

    print("Classes: ", *classes, sep=", ")
    print("Length of total dataset: ", len(dataset))

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for the custom dataset
    train_loader = DataLoader(
        CustomDataset(train_dataset), batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        CustomDataset(val_dataset), batch_size=batch_size, num_workers=0
    )

    return train_loader, valid_loader


def load_test_data(test_dir, preprocess, batch_size=BATCH_SIZE):
    test_dataset = ImageFolder(test_dir, transform=preprocess)

    # Create a DataLoader for the test data
    test_dataloader = DataLoader(
        CustomDataset(test_dataset), batch_size=batch_size, shuffle=False, num_workers=0
    )

    return test_dataloader
from configs import *
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, Dataset


def load_data(raw_dir, augmented_dir, external_dir, preprocess, batch_size=BATCH_SIZE):
    # Load the dataset using ImageFolder
    raw_dataset = ImageFolder(root=raw_dir, transform=preprocess)
    external_dataset = ImageFolder(root=external_dir, transform=preprocess)
    augmented_dataset = ImageFolder(root=augmented_dir, transform=preprocess)
    dataset = raw_dataset + external_dataset + augmented_dataset

    # Classes
    classes = augmented_dataset.classes

    print("Classes: ", *classes, sep=", ")
    print("Length of raw dataset: ", len(raw_dataset))
    print("Length of external dataset: ", len(external_dataset))
    print("Length of augmented dataset: ", len(augmented_dataset))
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

from configs import *
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader, Dataset


def load_data(original_dir, augmented_dir, preprocess):
    # Load the dataset using ImageFolder
    original_dataset = ImageFolder(root=original_dir, transform=preprocess)
    augmented_dataset = ImageFolder(root=augmented_dir, transform=preprocess)
    dataset = original_dataset + augmented_dataset

    print("Classes: ", *original_dataset.classes, sep = ' ')
    print("Length of original dataset: ", len(original_dataset))
    print("Length of augmented dataset: ", len(augmented_dataset))
    print("Length of total dataset: ", len(dataset))

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for the custom dataset
    train_loader = DataLoader(
        CustomDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        CustomDataset(val_dataset), batch_size=BATCH_SIZE, num_workers=0
    )
    
    return train_loader, valid_loader

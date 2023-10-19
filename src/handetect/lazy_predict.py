import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models import *
from torch.utils.tensorboard import SummaryWriter
from configs import *
import data_loader
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.utils import shuffle

def extract_features_labels(loader):
    data = []
    labels = []
    for inputs, labels_batch in loader:
        for img in inputs:
            data.append(img.view(-1).numpy())
        labels.extend(labels_batch.numpy())
    return np.array(data), np.array(labels)

def load_and_preprocess_data():
    train_loader, valid_loader = data_loader.load_data(
        RAW_DATA_DIR + str(TASK),
        AUG_DATA_DIR + str(TASK),
        EXTERNAL_DATA_DIR + str(TASK),
        preprocess,
    )
    return train_loader, valid_loader

def initialize_model_optimizer_scheduler(train_loader, valid_loader):
    model = MODEL.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    return model, criterion, optimizer, scheduler

# Load and preprocess data
train_loader, valid_loader = load_and_preprocess_data()

# Initialize the model, criterion, optimizer, and scheduler
model, criterion, optimizer, scheduler = initialize_model_optimizer_scheduler(train_loader, valid_loader)

# Extract features and labels
X_train, y_train = extract_features_labels(train_loader)
X_valid, y_valid = extract_features_labels(valid_loader)

# LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_valid, y_train, y_valid)

print("Models:", models)
print("Predictions:", predictions)







import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from data_loader import load_data, load_test_data
from configs import *
import numpy as np

torch.cuda.empty_cache()

# 


class MLP(nn.Module):
    def __init__(self, num_classes, num_models):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_classes * num_models, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.8),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def mlp_meta(num_classes, num_models):
    model = MLP(num_classes, num_models)
    return model


# Hyperparameters
input_dim = 3 * 224 * 224  # Modify this based on your input size
hidden_dim = 256
output_dim = NUM_CLASSES

# Create the data loaders using your data_loader functions50
train_loader, val_loader = load_data(COMBINED_DATA_DIR + "1", preprocess, BATCH_SIZE)
test_loader = load_test_data("data/test/Task 1", preprocess, BATCH_SIZE)

model_paths = [
    "output/checkpoints/bestsqueezenetSE3.pth",
    "output/checkpoints/EfficientNetB3WithDropout.pth",
    "output/checkpoints/MobileNetV2WithDropout2.pth",
]


# Define a function to load pretrained models
def load_pretrained_model(path, model):
    model.load_state_dict(torch.load(path))
    return model.to(DEVICE)


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


# Load pretrained models
model1 = load_pretrained_model(
    model_paths[0], SqueezeNet1_0WithSE(num_classes=NUM_CLASSES)
).to(DEVICE)
model2 = load_pretrained_model(
    model_paths[1], EfficientNetB3WithDropout(num_classes=NUM_CLASSES)
).to(DEVICE)
model3 = load_pretrained_model(
    model_paths[2], MobileNetV2WithDropout(num_classes=NUM_CLASSES)
).to(DEVICE)

models = [model1, model2, model3]

# Create the meta learner
meta_learner_model = mlp_meta(NUM_CLASSES, len(models)).to(DEVICE)
meta_optimizer = torch.optim.Adam(meta_learner_model.parameters(), lr=0.001)
meta_loss_fn = torch.nn.CrossEntropyLoss()

# Define the Cosine Annealing Learning Rate Scheduler
scheduler = CosineAnnealingLR(
    meta_optimizer, T_max=700
)  # T_max is the number of epochs for the cosine annealing.

# Define loss function and optimizer for the meta learner
criterion = nn.CrossEntropyLoss().to(DEVICE)

# Record learning rate
lr_hist = []

# Training loop
num_epochs = 160
for epoch in range(num_epochs):
    print("[Epoch: {}]".format(epoch + 1))
    print("Total number of batches: {}".format(len(train_loader)))
    for batch_idx, data in enumerate(train_loader, 0):
        print("Batch: {}".format(batch_idx + 1))
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1)

        # Forward pass through the three pretrained models
        features1 = model1(inputs)
        features2 = model2(inputs)
        features3 = model3(inputs)

        # Stack the features from the three models
        stacked_features = torch.cat((features1, features2, features3), dim=1).to(
            DEVICE
        )

        # Forward pass through the meta learner
        meta_output = meta_learner_model(stacked_features)

        # Compute the loss
        loss = cutmix_criterion(criterion, meta_output, targets_a, targets_b, lam)

        # Compute the accuracy
        _, predicted = torch.max(meta_output, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        # Backpropagation and optimization
        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()

        lr_hist.append(meta_optimizer.param_groups[0]["lr"])

        scheduler.step()

    print("Train Loss: {}".format(loss.item()))
    print("Train Accuracy: {}%".format(100 * correct / total))

    # Validation
    meta_learner_model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            features1 = model1(inputs)
            features2 = model2(inputs)
            features3 = model3(inputs)
            stacked_features = torch.cat((features1, features2, features3), dim=1).to(
                DEVICE
            )
            outputs = meta_learner_model(stacked_features)
            loss = criterion(outputs, labels)  # Use the validation loss
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Validation Loss: {}".format(val_loss / len(val_loader))
    )  # Calculate the average loss
    print("Validation Accuracy: {}%".format(100 * correct / total))


print("Finished Training")

# Test the ensemble
print("Testing the ensemble")
meta_learner_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        features1 = model1(inputs)
        features2 = model2(inputs)
        features3 = model3(inputs)
        stacked_features = torch.cat((features1, features2, features3), dim=1)
        outputs = meta_learner_model(stacked_features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    "Accuracy of the ensemble network on the test images: {}%".format(
        100 * correct / total
    )
)


# Plot the learning rate history

plt.plot(lr_hist)
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.title("Learning Rate History")
plt.show()


# Save the model
torch.save(meta_learner_model.state_dict(), "output/checkpoints/ensemble.pth")
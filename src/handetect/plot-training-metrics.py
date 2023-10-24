import pandas as pd
import matplotlib.pyplot as plt

# Load data from the CSV file
df = pd.read_csv('training_metrics.csv')

# Extract data
epochs = df['Epoch']
train_loss = df['Train Loss']
train_accuracy = df['Train Accuracy']
validation_loss = df['Validation Loss']
validation_accuracy = df['Validation Accuracy']

# Create subplots for loss and accuracy
plt.figure(figsize=(12, 5))

# Loss subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, validation_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label='Train Accuracy', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

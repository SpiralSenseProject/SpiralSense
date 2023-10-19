import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform
from configs import *

# Set the path to your dataset folder, where each subfolder represents a class
dataset_path = COMBINED_DATA_DIR + str(1)


# Function to load, resize, and convert images to grayscale
def load_resize_and_convert_to_gray(folder, target_size=(100, 100)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = imread(img_path, as_gray=True)
            img = resize(img, target_size, anti_aliasing=True)
            images.append(img)
    return images


# Load, resize, and convert images to grayscale from folders
X = []  # List to store images
y = []  # List to store corresponding labels

class_folders = os.listdir(dataset_path)
class_folders.sort()  # Sort the class folders to ensure consistent class ordering

for class_folder in class_folders:
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        images = load_resize_and_convert_to_gray(class_path)
        X.extend(images)
        y.extend([class_folder] * len(images))  # Assign labels based on folder name

# Convert data to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the parameter distributions for random search
param_dist = {
    "C": uniform(loc=0, scale=10),  # Randomly sample from [0, 10]
    "kernel": ["linear", "rbf", "poly"],
    "gamma": uniform(loc=0.001, scale=0.1),  # Randomly sample from [0.001, 0.1]
}

# Flatten the images to a 1D array
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create an SVM classifier
svm_classifier = svm.SVC()

# Perform Randomized Search with cross-validation
random_search = RandomizedSearchCV(
    svm_classifier,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42,
)

# Fit the Randomized Search on the training data
random_search.fit(X_train_flat, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(random_search.best_params_)

# Get the best SVM model with the tuned hyperparameters
best_svm_model = random_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_svm_model.predict(X_test_flat)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# You can also print other classification metrics like precision, recall, and F1-score
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

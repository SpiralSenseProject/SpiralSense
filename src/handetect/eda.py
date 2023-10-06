import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'

# Define the directory where your dataset is located
dataset_directory = 'data/train/combined/Task 1/'

# Create a list of class labels based on subdirectories in the dataset directory
class_labels = os.listdir(dataset_directory)

# Initialize lists to store data for EDA
num_samples_per_class = []
class_labels_processed = []

# Initialize an empty DataFrame to store image dimensions
image_dimensions_df = pd.DataFrame(columns=['Height', 'Width'])

# Initialize a dictionary to store a random sample of images from each class
sampled_images = {label: [] for label in class_labels}

# Iterate through class labels and count the number of samples per class
for label in class_labels:
    if label != ".DS_Store":
        class_directory = os.path.join(dataset_directory, label)
        num_samples = len(os.listdir(class_directory))
        num_samples_per_class.append(num_samples)
        class_labels_processed.append(label)
        
        # Extract image dimensions and add them to the DataFrame
        for image_file in os.listdir(class_directory):
            image_path = os.path.join(class_directory, image_file)
            image = plt.imread(image_path)
            height, width, _ = image.shape
            image_dimensions_df = image_dimensions_df._append({'Height': height, 'Width': width}, ignore_index=True)
            
            # Randomly sample 5 images from each class for visualization
            if len(sampled_images[label]) < 5:
                sampled_images[label].append(image)

# Create a Pandas DataFrame for EDA
eda_data = pd.DataFrame({'Class Label': class_labels_processed, 'Number of Samples': num_samples_per_class})

# Plot the number of samples per class
plt.figure(figsize=(10, 6))
sns.barplot(x='Class Label', y='Number of Samples', data=eda_data)
plt.title('Number of Samples per Class')
plt.xticks(rotation=45)
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.savefig('docs/eda/Number of Samples per Class.png')
plt.show()

# Calculate and plot the distribution of sample sizes (image dimensions)
plt.figure(figsize=(10, 6))
plt.scatter(image_dimensions_df['Width'], image_dimensions_df['Height'], alpha=0.5)
plt.title('Distribution of Sample Sizes (Image Dimensions)')
plt.xlabel('Width (Pixels)')
plt.ylabel('Height (Pixels)')
plt.savefig('docs/eda/Distribution of Sample Sizes (Image Dimensions).png')
plt.show()

# Plot a random sample of images from each class
for label, images in sampled_images.items():
    plt.figure(figsize=(15, 5))
    plt.suptitle(f'Random Sample of Images from Class: {label}')
    for i, image in enumerate(images, start=1):
        plt.subplot(1, 5, i)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Sample {i}')
    plt.savefig(f'docs/eda/Random Sample of Images from Class {label}.png')
    plt.show()

# Calculate and plot the correlation matrix for image dimensions
correlation_matrix = image_dimensions_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Image Dimensions')
plt.savefig('docs/eda/Correlation Matrix of Image Dimensions.png')
plt.show()

# Plot the distribution of image widths
plt.figure(figsize=(10, 6))
sns.histplot(image_dimensions_df['Width'], bins=20, kde=True)
plt.title('Distribution of Image Widths')
plt.xlabel('Width (Pixels)')
plt.ylabel('Frequency')
plt.savefig('docs/eda/Distribution of Image Widths.png')
plt.show()

# Plot the distribution of image heights
plt.figure(figsize=(10, 6))
sns.histplot(image_dimensions_df['Height'], bins=20, kde=True)
plt.title('Distribution of Image Heights')
plt.xlabel('Height (Pixels)')
plt.ylabel('Frequency')
plt.savefig('docs/eda/Distribution of Image Heights.png')
plt.show()

# Plot a table, each column is a test image, separate to 7 tables (one for each disease), each column have 4 rows, one is disease name, one is gradcam, one is lime, one is original image

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from configs import *
from sklearn.preprocessing import minmax_scale

plt.rcParams["font.family"] = "Times New Roman"

# Plot a table, each column is a test image, separate to 7 plot (one for each disease), each column have 4 rows, one is disease name, one is gradcam, one is lime, one is original image, the images are in 'docs/efficientnet/gradcam' and 'docs/efficientnet/lime' and 'data/test/Task 1'


def plot_table():
    diseases = CLASSES
    diseases.sort()
    # diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]
    print(diseases)
    fig, axs = plt.subplots(4, 14, figsize=(20, 10))
    fig.tight_layout()
    for i, disease in enumerate(diseases):
        # Create a new plot
        print("Processing", disease)
        axs[0, i].axis("off")
        axs[0, i].set_title(disease)
        axs[1, i].axis("off")
        axs[1, i].set_title("GradCAM")
        axs[2, i].axis("off")
        axs[2, i].set_title("LIME")
        axs[3, i].axis("off")
        axs[3, i].set_title("Original")
        # For each image in test folder, there are corresponding ones in gradcam folder and lime folder, plot it accordingly
        for j, image_path in enumerate(os.listdir(r"data\test\Task 1\{}".format(disease))):
            print("Processing", image_path)
            image_path = r"data\test\Task 1\{}\{}".format(disease, image_path)
            image_name = image_path.split(".")[0].split("\\")[-1]
            print("Processing", image_name)
            # Plot the original image
            image = cv2.imread(image_path, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axs[3, i].imshow(image)
            # Plot the gradcam image
            image = cv2.imread(
                f"docs/efficientnet/gradcam/{disease}/{image_name}.jpg", 1
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axs[1, i].imshow(image)
            # Plot the lime image
            image = cv2.imread(
                f"docs/efficientnet/lime/{disease}/{image_name}.jpg", 1
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axs[2, i].imshow(image)
            # # Plot the disease name
            # axs[0, i].text(0.5, 0.5, disease, horizontalalignment="center")
        plt.savefig("docs/efficientnet/table.png")
        plt.show()
    
if __name__ == "__main__":
    plot_table()
    
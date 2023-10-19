# Plot the gradcam pics of 7 classes from C:\Users\User\Documents\PISTEK\HANDETECT\docs\efficientnet\gradcam folder
# Each picture is named as <class_name>.jpg
# Usage: python plot-gradcam.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'

# Load the gradcam pics
gradcam_dir = r'C:\Users\User\Documents\PISTEK\HANDETECT\docs\efficientnet\gradcam'
gradcam_pics = []
for pic in os.listdir(gradcam_dir):
    gradcam_pics.append(cv2.imread(os.path.join(gradcam_dir, pic), 1))
    
# Plot the gradcam pics
plt.figure(figsize=(20, 20))
# Very tight layout
plt.tight_layout(pad=0.1)
for i, pic in enumerate(gradcam_pics):
    plt.subplot(3, 3, i + 1)
    plt.imshow(pic)
    plt.axis('off')
    plt.title(os.listdir(gradcam_dir)[i].split('.')[0], fontsize=13)
plt.savefig('docs/efficientnet/gradcam.jpg')
plt.show()


import os
import Augmentor
import shutil
from configs import *

tasks = ["1", "2", "3", "4", "5", "6"]

for task in tasks:
    # Loop through all folders in Task 1 and generate augmented images for each class
    for disease in os.listdir("data/train/raw/Task " + task):
        if disease != ".DS_Store":
            print("Augmenting images in class: ", disease, " in Task ", task)
            # Create a temp folder to combine the raw data and the external data
            if not os.path.exists(f"data/temp/Task {task}/{disease}/"):
                os.makedirs(f"data/temp/Task {task}/{disease}/")
            for file in os.listdir(f"data/train/raw/Task {task}/{disease}"):
                shutil.copy(
                    f"data/train/raw/Task {task}/{disease}/{file}",
                    f"data/temp/Task {task}/{disease}/{file}",
                )
            for file in os.listdir(f"data/train/external/Task {task}/{disease}"):
                shutil.copy(
                    f"data/train/external/Task {task}/{disease}/{file}",
                    f"data/temp/Task {task}/{disease}/{file}",
                )
            p = Augmentor.Pipeline(
                f"data/temp/Task {task}/{disease}",
                output_directory=f"{disease}/",
                save_format="png",
            )
            p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)
            p.flip_left_right(probability=0.8)
            p.zoom_random(probability=0.8, percentage_area=0.8)
            p.flip_top_bottom(probability=0.8)
            p.random_brightness(probability=0.8, min_factor=0.5, max_factor=1.5)
            p.random_contrast(probability=0.8, min_factor=0.5, max_factor=1.5)
            p.random_color(probability=0.8, min_factor=0.5, max_factor=1.5)
            # Generate 100 - total of original images so that the total number of images in each class is 100
            p.sample(100 - len(p.augmentor_images))
            # Move the folder to data/train/Task 1/augmented
            # Create the folder if it does not exist
            if not os.path.exists(f"data/train/augmented/Task {task}/"):
                os.makedirs(f"data/train/augmented/Task {task}/")
            # Move all images in the data/train/Task 1/i folder to data/train/Task 1/augmented/i
            os.rename(
                f"data/temp/Task {task}/{disease}/{disease}",
                f"data/train/augmented/Task {task}/{disease}",
            )
            # Rename all the augmented images to [01, 02, 03]
            number = 0
            for file in os.listdir(f"data/train/augmented/Task {task}/{disease}"):
                number = int(number) + 1
                if len(str(number)) == 1:
                    number = "0" + str(number)
                os.rename(
                    f"data/train/augmented/Task {task}/{disease}/{file}",
                    f"data/train/augmented/Task {task}/{disease}/{number}.png",
                )

import os
import Augmentor
import shutil
from configs import *
import uuid

tasks = ["1", "2", "3", "4", "5", "6"]
num_of_images = 100

shutil.rmtree(TEMP_DATA_DIR, ignore_errors=True)


for task in ["1"]:
    shutil.rmtree(AUG_DATA_DIR + task, ignore_errors=True)
    # Loop through all folders in Task 1 and generate augmented images for each class
    for class_label in [
        "Alzheimer Disease",
        "Cerebral Palsy",
        "Dystonia",
        "Essential Tremor",
        "Healthy",
        "Huntington Disease",
        "Parkinson Disease",
    ]:
        if class_label != ".DS_Store":
            print("Augmenting images in class: ", class_label, " in Task ", task)
            # Create a temp folder to combine the raw data and the external data
            if not os.path.exists(f"{TEMP_DATA_DIR}{task}/{class_label}/"):
                os.makedirs(f"{TEMP_DATA_DIR}{task}/{class_label}/")
            if os.path.exists(f"{RAW_DATA_DIR}{task}/{class_label}"):
                for file in os.listdir(f"{RAW_DATA_DIR}{task}/{class_label}"):
                    shutil.copy(
                        f"{RAW_DATA_DIR}{task}/{class_label}/{file}",
                        f"{TEMP_DATA_DIR}{task}/{class_label}/{str(uuid.uuid4())}.png",
                    )
            if os.path.exists(f"{EXTERNAL_DATA_DIR}{task}/{class_label}"):
                for file in os.listdir(f"{EXTERNAL_DATA_DIR}{task}/{class_label}"):
                    shutil.copy(
                        f"{EXTERNAL_DATA_DIR}{task}/{class_label}/{file}",
                        f"{TEMP_DATA_DIR}{task}/{class_label}/{str(uuid.uuid4())}.png",
                    )
            p = Augmentor.Pipeline(
                f"{TEMP_DATA_DIR}{task}/{class_label}",
                output_directory=f"{class_label}/",
                save_format="png",
            )
            p.flip_left_right(probability=0.8)
            p.zoom_random(probability=0.8, percentage_area=0.8)
            p.flip_top_bottom(probability=0.8)
            p.random_brightness(probability=0.8, min_factor=0.5, max_factor=1.5)
            p.random_contrast(probability=0.8, min_factor=0.5, max_factor=1.5)
            p.random_color(probability=0.8, min_factor=0.5, max_factor=1.5)
            p.rotate_random_90(probability=0.8)
            p.sample(num_of_images - len(p.augmentor_images))
            # Move the folder to data/train/Task 1/augmented
            # Create the folder if it does not exist
            if not os.path.exists(f"{AUG_DATA_DIR}{task}/"):
                os.makedirs(f"{AUG_DATA_DIR}{task}/")
            # Move all images in the data/train/Task 1/i folder to data/train/Task 1/augmented/i
            os.rename(
                f"{TEMP_DATA_DIR}{task}/{class_label}/{class_label}",
                f"{AUG_DATA_DIR}{task}/{class_label}",
            )
            # Rename all the augmented images to [01, 02, 03]
            number = 0
            for file in os.listdir(f"{AUG_DATA_DIR}{task}/{class_label}"):
                number = int(number) + 1
                if len(str(number)) == 1:
                    number = "0" + str(number)
                os.rename(
                    f"{AUG_DATA_DIR}{task}/{class_label}/{file}",
                    f"{AUG_DATA_DIR}{task}/{class_label}/{number}.png",
                )

shutil.rmtree(TEMP_DATA_DIR)

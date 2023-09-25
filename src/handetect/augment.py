import os
import Augmentor

tasks = ["1", "2", "3", "4", "5", "6"]

for task in tasks:
    # Loop through all folders in Task 1 and generate augmented images for each class
    for i in os.listdir("data/train/raw/Task " + task):
        if i != ".DS_Store":
            print("Augmenting images in class: ", i)
            p = Augmentor.Pipeline(f"data/train/raw/Task {task}/{i}", output_directory=i, save_format="png")
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
                f"data/train/raw/Task {task}/{i}/{i}",
                f"data/train/augmented/Task {task}/{i}",
            )
            # Rename all the augmented images to [01, 02, 03]
            number = 0
            for j in os.listdir(f"data/train/augmented/Task {task}/{i}"):
                number = int(number) + 1
                if len(str(number)) == 1:
                    number = "0" + str(number)
                os.rename(
                    f"data/train/augmented/Task {task}/{i}/{j}",
                    f"data/train/augmented/Task {task}/{i}/{number}.png",
                )



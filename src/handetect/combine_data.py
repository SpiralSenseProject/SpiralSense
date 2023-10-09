# Copy the all the data from external, augmented and raw folders to combined folder
import os
import shutil
import uuid

from configs import *

shutil.rmtree(COMBINED_DATA_DIR, ignore_errors=True)

for disease in CLASSES:
    # check if the original folder exists
    if os.path.exists(RAW_DATA_DIR + "1/" + disease):
        print("Copying raw data for disease: ", disease)
        if not os.path.exists(COMBINED_DATA_DIR + "1/" + disease):
            os.makedirs(COMBINED_DATA_DIR + "1/" + disease)
        for file in os.listdir(RAW_DATA_DIR + "1/" + disease):
            random_name = str(uuid.uuid4()) + ".png"
            shutil.copy(
                RAW_DATA_DIR + "1/" + disease + "/" + file,
                COMBINED_DATA_DIR + "1/" + disease + "/" + random_name,
            )

    if os.path.exists(EXTERNAL_DATA_DIR + "1/" + disease):
        print("Copying external data for disease: ", disease)
        if not os.path.exists(COMBINED_DATA_DIR + "1/" + disease):
            os.makedirs(COMBINED_DATA_DIR + "1/" + disease)
        for file in os.listdir(EXTERNAL_DATA_DIR + "1/" + disease):
            random_name = str(uuid.uuid4()) + ".png"
            shutil.copy(
                EXTERNAL_DATA_DIR + "1/" + disease + "/" + file,
                COMBINED_DATA_DIR + "1/" + disease + "/" + random_name,
            )

    if os.path.exists(AUG_DATA_DIR + "1/" + disease):
        print("Copying augmented data for disease: ", disease)
        if not os.path.exists(COMBINED_DATA_DIR + "1/" + disease):
            os.makedirs(COMBINED_DATA_DIR + "1/" + disease)
        for file in os.listdir(AUG_DATA_DIR + "1/" + disease):
            random_name = str(uuid.uuid4()) + ".png"
            shutil.copy(
                AUG_DATA_DIR + "1/" + disease + "/" + file,
                COMBINED_DATA_DIR + "1/" + disease + "/" + random_name,
            )

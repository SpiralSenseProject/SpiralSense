# Take 10% of the data in data\train\combined\Task 1\<class> and move it to data\test\Task 1\<class>

import os
import shutil
import uuid

from configs import *

shutil.rmtree(TEST_DATA_DIR + "1/", ignore_errors=True)

for disease in CLASSES:
    # check if the original folder exists
    if os.path.exists(COMBINED_DATA_DIR + "1/" + disease):
        print("Splitting data for disease: ", disease)
        if not os.path.exists(TEST_DATA_DIR + "1/" + disease):
            os.makedirs(TEST_DATA_DIR + "1/" + disease)
        files = os.listdir(COMBINED_DATA_DIR + "1/" + disease)
        files.sort()
        for file in files[: int(len(files) * 0.1)]:
            shutil.move(
                COMBINED_DATA_DIR + "1/" + disease + "/" + file,
                TEST_DATA_DIR + "1/" + disease + "/" + file,
            )
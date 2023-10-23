from gradio_client import Client
import time
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
from configs import *
from PIL import Image

client = Client("https://cycool29-handetect.hf.space/")

list_of_times = []


rcParams["font.family"] = "Times New Roman"

# Load the model
model = MODEL.to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

for disease in CLASSES:
    print("Processing", disease)
    for image_path in os.listdir(r"data\test\Task 1\{}".format(disease)):
        # print("Processing", image_path)
        image_path = r"data\test\Task 1\{}\{}".format(disease, image_path)
        start_time = time.time()
        result = client.predict(
                        image_path,	
                        api_name="/predict"
        )
        time_taken = time.time() - start_time
        list_of_times.append(time_taken)
        print("Time taken:", time_taken)

        # Log to csv
        with open('log.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([disease])
            writer.writerow([image_path])
            writer.writerow([time_taken])
            

print("Average time taken:", sum(list_of_times)/len(list_of_times))
print("Max time taken:", max(list_of_times))
print("Min time taken:", min(list_of_times))
print("Total time taken:", sum(list_of_times))
print("Median time taken:", sorted(list_of_times)[len(list_of_times)//2])

# Plot the histogram
plt.hist(list_of_times, bins=10)
plt.xlabel("Time taken (s)")
plt.ylabel("Frequency")
plt.title("Time taken to process each image")
plt.savefig("docs/efficientnet/time_taken_for_web.png")


# Now is local
list_of_times = []

for disease in CLASSES:
    print("Processing", disease)
    for image_path in os.listdir(r"data\test\Task 1\{}".format(disease)):
        # print("Processing", image_path)
        image_path = r"data\test\Task 1\{}\{}".format(disease, image_path)
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0)
        image = image.to(DEVICE)
        output = model(image)
        time_taken = time.time() - start_time
        list_of_times.append(time_taken)
        print("Time taken:", time_taken)

        # Log to csv
        with open('log.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([disease])
            writer.writerow([image_path])
            writer.writerow([time_taken])


print("Average time taken local:", sum(list_of_times)/len(list_of_times))
print("Max time taken local:", max(list_of_times))
print("Min time taken local:", min(list_of_times))
print("Total time taken local:", sum(list_of_times))
print("Median time taken local:", sorted(list_of_times)[len(list_of_times)//2])

# Plot the histogram
plt.hist(list_of_times, bins=10)
plt.xlabel("Time taken (s) local")
plt.ylabel("Frequency local")
plt.title("Time taken to process each image local")
plt.savefig("docs/efficientnet/time_taken_for_local.png")
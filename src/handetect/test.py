import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from configs import *

# Load a pre-trained model (e.g., VGG16)
model = MODEL.to(DEVICE)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

# Prepare an initial image (e.g., a random noise image)
image = torch.randn(1, 3, 224, 224, requires_grad=True).cuda()

# Define a loss function to maximize a specific layer or neuron's activation
loss_function = torch.nn.CrossEntropyLoss()

# Create an optimizer (e.g., Adam) for updating the image
optimizer = optim.Adam([image], lr=0.01)

# Optimization loop
for _ in range(1000):
    optimizer.zero_grad()
    output = model(image)
    loss = -output[0, 'Healthy']  # Maximize the activation of a specific class
    loss.backward()
    optimizer.step()

# Visualize the optimized image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Convert the tensor to an image
image = transforms.ToPILImage()(image.squeeze())

# Display the generated image
plt.imshow(image)
plt.axis('off')
plt.show()

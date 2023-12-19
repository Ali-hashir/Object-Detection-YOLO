import torch
from PIL import Image
import time

# Load the image
image_path = 'yolov5\WhatsApp Image 2023-12-18 at 11.20.06_91831e86.jpg'
image = Image.open(image_path)

# Load a pre-trained YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Measure inference time
start_time = time.time()
results = model(image)
end_time = time.time()

# Display results
results.show()

# Print inference time
print(f"Inference Time: {end_time - start_time} seconds")

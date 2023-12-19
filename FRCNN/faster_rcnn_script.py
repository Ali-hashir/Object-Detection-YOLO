
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import time

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to inference mode

def load_input_image(image_path):
    # Load an image and preprocess it
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)
    return image

def predict_image(image_path):
    image = load_input_image(image_path)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        prediction = model([image])
    end_time = time.time()

    print(f"Inference Time: {end_time - start_time} seconds")
    return prediction

# Replace 'path_to_your_image.jpg' with your image file
prediction = predict_image('FRCNN\WhatsApp Image 2023-12-18 at 11.20.06_91831e86.jpg')
print(prediction)

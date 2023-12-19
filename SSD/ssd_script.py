
import torchvision
import torch
from torchvision import transforms
from PIL import Image
import time
import cv2
import numpy as np

# Load a pre-trained SSD model
model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()  # Set the model to inference mode

def load_input_image(image_path):
    # Load an image and preprocess it
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((300, 300)), 
                                    transforms.ToTensor()])
    image = transform(image)
    return image

def draw_predictions(image_np, prediction):
    # Draw the predictions on the image
    for element in range(len(prediction[0]['boxes'])):
        boxes = prediction[0]['boxes'][element].cpu().numpy()
        score = np.round(prediction[0]['scores'][element].cpu().numpy(), decimals=4)
        label = prediction[0]['labels'][element].cpu().numpy()

        if score > 0.5:  # You can adjust the threshold here
            cv2.rectangle(image_np, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 255, 0), 2)
            cv2.putText(image_np, f'{label} {score}', (int(boxes[0]), int(boxes[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image_np

def predict_image(image_path):
    image = load_input_image(image_path)
    original_image = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        prediction = model([image])
    end_time = time.time()

    print(f"Inference Time: {end_time - start_time} seconds")
    processed_image = draw_predictions(original_image, prediction)
    return processed_image

# Replace 'path_to_your_image.jpg' with your image file
processed_image = predict_image('yolov5\WhatsApp Image 2023-12-11 at 16.00.29_3ea35051.jpg')

# Save or display the result
cv2.imshow('SSD Predictions', processed_image)
cv2.waitKey(0)  # Wait for a key press to close the image window
cv2.destroyAllWindows()

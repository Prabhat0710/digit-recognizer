import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

model = DigitCNN()
model.load_state_dict(torch.load('model/digit_model.pth'))
model.eval()

def predict(image):
    if image is None:
        return {"error": 1.0}
    img = Image.fromarray(image).convert('L')

    img_array = np.array(img)
    img_array = (img_array < 180).astype(np.uint8) * 255
    img = Image.fromarray(img_array)
    img = img.filter(ImageFilter.MedianFilter(size=3))

    bbox = img.getbbox()
    if bbox is None:
        return "No digit detected"
    img = img.crop(bbox)
    img = ImageOps.expand(img, border=10, fill=0)
    img = img.resize((28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        predicted = probs.argmax().item()
        confidence = probs[predicted].item()

    return {str(i): float(probs[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload handwritten digit"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="🧠 Handwritten Digit Recognizer",
    description="Upload an image of a handwritten digit (0–9). Model: CNN trained on MNIST — 99.67% accuracy.",
    examples=[]
)

demo.launch()
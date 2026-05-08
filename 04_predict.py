import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── Load model ────────────────────────────────────
model = DigitCNN()
model.load_state_dict(torch.load('model/digit_model.pth'))
model.eval()
print("Model loaded.")

# ── Predict ───────────────────────────────────────
def predict(image_path):
    img = Image.open(image_path).convert('L')

    inverted = ImageOps.invert(img)
    bbox = inverted.getbbox()
    img = img.crop(bbox)
    img = ImageOps.expand(img, border=10, fill=255)
    img = img.resize((28, 28))
    img = ImageOps.invert(img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        predicted = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {predicted}  |  Confidence: {confidence*100:.1f}%")
    plt.axis('off')
    plt.show()

    return predicted, confidence

# ── Run ───────────────────────────────────────────
path = input("Enter image path: ").strip()
digit, conf = predict(path)
print(f"\nPredicted digit : {digit}")
print(f"Confidence      : {conf*100:.1f}%")
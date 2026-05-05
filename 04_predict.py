import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# ── Model (same architecture) ─────────────────────
class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.net(x)

# ── Load saved weights ────────────────────────────
model = DigitNet()
model.load_state_dict(torch.load('model/digit_model.pth'))
model.eval()
print("Model loaded.")

# ── Predict function ──────────────────────────────
def predict(image_path):
    img = Image.open(image_path).convert('L')  # grayscale

    # Crop to digit
    inverted = ImageOps.invert(img)            # invert to find bbox
    bbox = inverted.getbbox()                  # find digit bounds
    img = img.crop(bbox)                       # crop original

    # Add padding
    img = ImageOps.expand(img, border=60, fill=255)

    # Resize
    img = img.resize((28, 28))

    # Convert to MNIST style: white digit on BLACK background
    img = ImageOps.invert(img)                 # NOW invert AFTER resize

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST exact mean/std
    ])

    tensor = transform(img).unsqueeze(0)       # add batch dimension

    with torch.no_grad():
        output = model(tensor)
        predicted = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted].item()

    # Show the image
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
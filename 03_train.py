import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ── Data + Augmentation ───────────────────────────
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data', train=True,  download=True, transform=train_transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# ── CNN Model ─────────────────────────────────────
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28, 32 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 28x28 → 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 → 14x14, 64 filters
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14 → 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 7x7 → 7x7, 128 filters
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),                             # prevent overfitting
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model     = DigitCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training ──────────────────────────────────────
epochs = 10
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}  |  Loss: {avg_loss:.4f}")

# ── Test Accuracy ─────────────────────────────────
model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for images, labels in test_loader:
        output    = model(images)
        predicted = output.argmax(dim=1)
        correct  += (predicted == labels).sum().item()
        total    += labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# ── Save ──────────────────────────────────────────
torch.save(model.state_dict(), 'model/digit_model.pth')
print("Model saved → model/digit_model.pth")

# ── Loss curve ────────────────────────────────────
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.title('CNN Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
plt.show()
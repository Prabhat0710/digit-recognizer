import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --Data-----
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root='./data', train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# --Model-----
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
    
model = DigitNet()
criterion = nn.CrossEntropyLoss() # it will measure how wrong the model is
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # it will adjust weights

# --Training-----
epochs = 10
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad() # Reset gradients
        output = model(images) # forward pass 
        loss = criterion(output, labels) # claculate loss
        loss.backward() # backprop
        optimizer.step() #update weights
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epoch} | Loss: {avg_loss:.4f}")

# --Test Accuracy-----
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        predicted = output.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")

# --Save Model----
torch.save(model.state_dict(), 'model/digit_model.pth')
print("Model saved to model/digit_model.pth")

# --Plot Loss Curve-----
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')
plt.show()
print("Loss curve saved.")
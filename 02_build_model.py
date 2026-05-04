import torch
import torch.nn as nn

class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),           # 28x28 image → 784 numbers (one long list)
            nn.Linear(784, 128),    # 784 inputs → 128 neurons
            nn.ReLU(),              # activation (explained below)
            nn.Linear(128, 64),     # 128 → 64 neurons
            nn.ReLU(),
            nn.Linear(64, 10)       # 64 → 10 outputs (digits 0-9)
        )

    def forward(self, x):
        return self.net(x)

# Test it works
model = DigitNet()
print(model)

# Feed fake image through — should output 10 numbers
fake_image = torch.randn(1, 1, 28, 28)  # batch=1, channel=1, 28x28
output = model(fake_image)
print(f"Output shape: {output.shape}")  # should be torch.Size([1, 10])
print(f"Raw output: {output}")
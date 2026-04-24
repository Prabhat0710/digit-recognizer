import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Download MNIST dataset
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print(f"Training samples : {len(train_data)}")   # 60,000
print(f"Test samples     : {len(test_data)}")     # 10,000
print(f"Image shape      : {train_data[0][0].shape}")  # torch.Size([1, 28, 28])

# Visualize first 12 digits
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
for i, ax in enumerate(axes.flat):
    image, label = train_data[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.suptitle("MNIST Sample Images", fontsize=14)
plt.tight_layout()
plt.savefig("mnist_samples.png")
plt.show()
print("Saved: mnist_samples.png")
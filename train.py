import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

print("🚀 Training Started...")

# Dataset path
data_dir = "dataset"

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = ImageFolder(root=data_dir, transform=transform)

print("📊 Total Dataset Size:", len(dataset))

if len(dataset) == 0:
    print("❌ Dataset empty! Check folders.")
    exit()

# 🔥 LIMIT DATASET (IMPORTANT FOR SPEED)
max_samples = 2000   # change if needed
if len(dataset) > max_samples:
    dataset, _ = random_split(dataset, [max_samples, len(dataset) - max_samples])
    print(f"⚡ Using only {max_samples} samples for fast training")

# Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, _ = random_split(dataset, [train_size, test_size])

# 🔥 Smaller batch = faster on CPU
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Model (NO DOWNLOAD)
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 🔥 Reduce epochs for speed
epochs = 2

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i, (images, labels) in enumerate(train_loader):

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 Progress print
        if i % 50 == 0:
            print(f"Epoch {epoch+1} | Step {i} | Loss: {loss.item():.4f}")

    print(f"✅ Epoch {epoch+1} Completed | Total Loss: {total_loss:.4f}\n")

# Save model
torch.save(model.state_dict(), "deepfake_model.pth")

print("🎉 Model Training Completed & Saved!")
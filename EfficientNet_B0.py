import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
import os

from densenet import DenseNet121

# ============================================================
# 1️⃣ Cek GPU
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device yang digunakan:", device)

# ============================================================
# 2️⃣ Preprocessing Data
# ============================================================
data_dir = "dataset"  # ubah sesuai nama folder dataset kamu
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = ImageFolder(train_dir, transform=transform)
val_data = ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print(f"Jumlah data training: {len(train_data)}")
print(f"Jumlah data validasi: {len(val_data)}")

# ============================================================
# 3️⃣ Load Model DenseNet121
# ============================================================
num_classes = 2
model = DenseNet121(in_channels=1, num_classes=num_classes)
model = model.to(device)
print("\nModel DenseNet121 berhasil dibuat!\n")

# ============================================================
# 4️⃣ Definisi Loss dan Optimizer
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# 5️⃣ Proses Training
# ============================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%")

        validate_model(model, val_loader)

# ============================================================
# 6️⃣ Fungsi Validasi
# ============================================================
def validate_model(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validasi Akurasi: {100 * correct / total:.2f}%")

# ============================================================
# 7️⃣ Jalankan Training
# ============================================================
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5)

# ============================================================
# 8️⃣ Simpan Model
# ============================================================
torch.save(model.state_dict(), "DenseNet121_ChestMNIST.pth")
print("Model telah disimpan: DenseNet121_ChestMNIST.pth")

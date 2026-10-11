
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

DATA_DIR = Path.home() / "Downloads/dogvcat/train/train"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20

random.seed(SEED)
torch.manual_seed(SEED)

# Gather image paths and labels.
samples = []
for path in DATA_DIR.glob("*.jpg"):
    name = path.name.lower()
    if name.startswith("cat."):
        samples.append((path, 0))
    elif name.startswith("dog."):
        samples.append((path, 1))

random.shuffle(samples)
split = int(len(samples) * 0.8)
train_samples = samples[:split]
val_samples = samples[split:]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


class AnimalDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, label


train_dataset = AnimalDataset(train_samples, train_transform)
val_dataset = AnimalDataset(val_samples, val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


device = torch.device("cpu")
model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if step % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, "
                  f"batch {step}/{len(train_loader)}")

    train_loss = total_loss / total
    train_acc = correct / total
    val_loss, val_acc = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS}: "
        f"train loss={train_loss:.4f}, "
        f"train accuracy={train_acc:.2%}, "
        f"validation loss={val_loss:.4f}, "
        f"validation accuracy={val_acc:.2%}"
    )

torch.save(model.state_dict(), OUTPUT_DIR / "small_cnn.pt")

epochs = range(1, EPOCHS + 1)
plt.figure()
plt.plot(epochs, history["train_acc"], label="Training")
plt.plot(epochs, history["val_acc"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy.png")
plt.close()

plt.figure()
plt.plot(epochs, history["train_loss"], label="Training")
plt.plot(epochs, history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss.png")
plt.close()

print("Saved model and graphs in:", OUTPUT_DIR.resolve())


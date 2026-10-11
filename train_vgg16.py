
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# Paths and settings
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path.home() / "Downloads/dogvcat/train/train"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cpu")
print(f"Using device: {device}", flush=True)

# Preprocessing for pretrained ImageNet weights
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Dataset for filenames such as cat.0.jpg and dog.0.jpg
class CatDogDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = 0 if path.name.startswith("cat.") else 1

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Build a deterministic 80/20 split
samples = sorted(
    path for path in DATA_DIR.glob("*.jpg")
    if path.name.startswith(("cat.", "dog."))
)

random.shuffle(samples)
split = int(0.8 * len(samples))
train_paths = samples[:split]
val_paths = samples[split:]

if len(samples) == 0:
    raise RuntimeError(f"No cat/dog images found in {DATA_DIR}")

train_dataset = CatDogDataset(train_paths, train_transform)
val_dataset = CatDogDataset(val_paths, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Total images: {len(samples)}", flush=True)
print(f"Training images: {len(train_dataset)}", flush=True)
print(f"Validation images: {len(val_dataset)}", flush=True)
print("Class mapping: cat=0, dog=1", flush=True)

# Load pretrained VGG-16 weights
print("Loading pretrained VGG-16; downloading weights if needed...", flush=True)
weights = models.VGG16_Weights.IMAGENET1K_V1
model = models.vgg16(weights=weights)

# Freeze convolutional feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False

# Replace final ImageNet classifier layer with two output classes
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features, 2
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
)

best_val_accuracy = 0.0
best_model_path = OUTPUT_DIR / "vgg16_best.pt"

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = 100 * train_correct / train_total
    val_loss = val_loss_sum / val_total
    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%",
        flush=True,
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved: {best_model_path}", flush=True)

print(f"Best validation accuracy: {best_val_accuracy:.2f}%", flush=True)
print("VGG-16 experiment finished.", flush=True)


import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

random.seed(42)
torch.manual_seed(42)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path.home() / "Downloads/dogvcat/train/train"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CACHE_DIR = OUTPUT_DIR / "vgg16_feature_cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class CatDogDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = 0 if path.name.startswith("cat.") else 1
        with Image.open(path) as image:
            image = transform(image.convert("RGB"))
        return image, label

samples = sorted(
    path for path in DATA_DIR.glob("*.jpg")
    if path.name.startswith(("cat.", "dog."))
)
random.shuffle(samples)
split = int(0.8 * len(samples))
train_paths, val_paths = samples[:split], samples[split:]

print(f"Training images: {len(train_paths)}", flush=True)
print(f"Validation images: {len(val_paths)}", flush=True)

weights = models.VGG16_Weights.IMAGENET1K_V1
model = models.vgg16(weights=weights)
features_model = model.features.to(device)
features_model.eval()

for parameter in features_model.parameters():
    parameter.requires_grad = False

def extract_features(paths, cache_name):
    cache_file = CACHE_DIR / f"{cache_name}.pt"

    if cache_file.exists():
        print(f"Loading cached {cache_name} features...", flush=True)
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    loader = DataLoader(
        CatDogDataset(paths),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    all_features = []
    all_labels = []
    total = len(paths)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            features = features_model(images).flatten(1)
            all_features.append(features.cpu())
            all_labels.append(labels)

            if (batch_index + 1) % 100 == 0 or batch_index + 1 == len(loader):
                done = min((batch_index + 1) * BATCH_SIZE, total)
                print(
                    f"{cache_name}: {done}/{total} images processed",
                    flush=True,
                )

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    torch.save((features, labels), cache_file)
    print(f"Saved cache: {cache_file}", flush=True)
    return features, labels

train_features, train_labels = extract_features(train_paths, "train")
val_features, val_labels = extract_features(val_paths, "validation")

# Train a smaller classifier on the frozen VGG-16 features
classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),
).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

best_accuracy = 0.0
best_path = OUTPUT_DIR / "vgg16_cached_best.pt"

train_data = DataLoader(
    list(zip(train_features, train_labels)),
    batch_size=64,
    shuffle=True,
)
val_data = DataLoader(
    list(zip(val_features, val_labels)),
    batch_size=64,
    shuffle=False,
)

for epoch in range(EPOCHS):
    classifier.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for features, labels in train_data:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    classifier.eval()
    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for features, labels in val_data:
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_accuracy = 100 * correct / total
    train_loss = loss_sum / total
    val_accuracy = 100 * val_correct / val_total
    val_loss = val_loss_sum / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%",
        flush=True,
    )

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
            "classifier_state_dict": classifier.state_dict(),
            "class_to_idx": {"cat": 0, "dog": 1},
            "image_size": IMAGE_SIZE,
            "best_val_accuracy": best_accuracy,
        }, best_path)
        print(f"Saved best checkpoint: {best_path}", flush=True)

print(f"Best validation accuracy: {best_accuracy:.2f}%", flush=True)
print("Cached VGG-16 experiment complete.", flush=True)

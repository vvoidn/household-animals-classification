from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = Path.home() / "Downloads/dogvcat/train/train"
MODEL_PATH = Path("outputs/small_cnn.pt")
OUTPUT_DIR = Path("outputs")
IMAGE_SIZE = 128
BATCH_SIZE = 32
SEED = 42

# Re-create the same validation split as training
random.seed(SEED)
samples = []

for path in DATA_DIR.glob("*.jpg"):
    name = path.name.lower()
    if name.startswith("cat."):
        samples.append((path, 0))
    elif name.startswith("dog."):
        samples.append((path, 1))

random.shuffle(samples)
split = int(len(samples) * 0.8)
val_samples = samples[split:]

transform = transforms.Compose([
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


if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

dataset = AnimalDataset(val_samples, transform)
loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

device = torch.device("cpu")
model = SmallCNN().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

actual = []
predicted = []

print(f"Evaluating {len(dataset)} validation images...")

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()

        actual.extend(labels.tolist())
        predicted.extend(predictions.tolist())

print("\nClassification report:")
print(classification_report(
    actual,
    predicted,
    labels=[0, 1],
    target_names=["Cat", "Dog"],
    digits=4,
    zero_division=0,
))

matrix = confusion_matrix(actual, predicted, labels=[0, 1])
print("Confusion matrix (rows = actual, columns = predicted):")
print(matrix)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Cat", "Dog"],
)
display.plot(cmap="Blues", values_format="d")
plt.title("Cat vs Dog - Validation Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print(f"\nOverall accuracy: {sum(a == p for a, p in zip(actual, predicted)) / len(actual):.2%}")
print(f"Confusion matrix saved to: {OUTPUT_DIR / 'confusion_matrix.png'}")

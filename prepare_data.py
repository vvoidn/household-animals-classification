from pathlib import Path
from collections import Counter
from PIL import Image
import random

DATA_DIR = Path.home() / "Downloads/dogvcat/train/train"
SEED = 42

random.seed(SEED)

images = []
labels = []

for path in DATA_DIR.glob("*.jpg"):
    name = path.name.lower()

    if name.startswith("cat."):
        images.append(path)
        labels.append(0)
    elif name.startswith("dog."):
        images.append(path)
        labels.append(1)

print("Images found:", len(images))
print("Cats:", labels.count(0))
print("Dogs:", labels.count(1))

# Shuffle image paths and labels together.
combined = list(zip(images, labels))
random.shuffle(combined)

split = int(len(combined) * 0.8)
train_data = combined[:split]
val_data = combined[split:]

print("Training images:", len(train_data))
print("Validation images:", len(val_data))

# Check a small sample for corrupted images.
bad_images = []

for path, _ in train_data[:100]:
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        bad_images.append(path)

print("Corrupted images in checked sample:", len(bad_images))


from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMAGE_SIZE = 224
MODEL_PATH = Path("outputs/vgg16_cached_best.pt")

if len(sys.argv) < 2:
    print("Usage: python predict_vgg16.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print(f"Image not found: {image_path}")
    sys.exit(1)

if not MODEL_PATH.is_file():
    print(f"Trained VGG-16 checkpoint not found: {MODEL_PATH}")
    sys.exit(1)

device = torch.device("cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

# Re-create the trained classifier
classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),
)

classifier.load_state_dict(checkpoint["classifier_state_dict"])
classifier.eval()

# Load pretrained ImageNet VGG-16 features from the local cache
weights_path = (
    Path.home()
    / ".cache/torch/hub/checkpoints/vgg16-397923af.pth"
)

if not weights_path.is_file():
    raise FileNotFoundError(
        f"Pretrained VGG-16 weights not found: {weights_path}"
    )

vgg = models.vgg16(weights=None)
pretrained = torch.load(
    weights_path,
    map_location=device,
    weights_only=True,
)
vgg.load_state_dict(pretrained)
features_model = vgg.features.to(device)
features_model.eval()

# Match the preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

with Image.open(image_path) as image:
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

# Extract VGG-16 features, then classify them
with torch.no_grad():
    features = features_model(image_tensor).flatten(1)
    logits = classifier(features)
    probabilities = torch.softmax(logits, dim=1)[0]

labels = ["Cat", "Dog"]
predicted_class = probabilities.argmax().item()

print(f"\nImage: {image_path.name}")
print(f"Prediction: {labels[predicted_class]}")
print(f"Cat probability: {probabilities[0].item():.2%}")
print(f"Dog probability: {probabilities[1].item():.2%}")

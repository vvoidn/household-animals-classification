from pathlib import Path
import sys

import torch
from torch import nn
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 128
MODEL_PATH = Path("outputs/small_cnn.pt")

# Use the same CNN architecture as training
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


if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print(f"Image not found: {image_path}")
    sys.exit(1)

if not MODEL_PATH.is_file():
    print(f"Trained model not found: {MODEL_PATH}")
    sys.exit(1)

device = torch.device("cpu")
model = SmallCNN().to(device)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

with Image.open(image_path) as image:
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]

predicted_class = probabilities.argmax().item()
labels = ["Cat", "Dog"]

print(f"\nImage: {image_path.name}")
print(f"Prediction: {labels[predicted_class]}")
print(f"Cat probability: {probabilities[0].item():.2%}")
print(f"Dog probability: {probabilities[1].item():.2%}")

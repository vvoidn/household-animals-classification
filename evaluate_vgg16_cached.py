
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

# Paths
OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / "vgg16_cached_best.pt"
VALIDATION_CACHE = OUTPUT_DIR / "vgg16_feature_cache" / "validation.pt"

# Device
device = torch.device("cpu")

# Check required files
if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

if not VALIDATION_CACHE.exists():
    raise FileNotFoundError(f"Validation cache not found: {VALIDATION_CACHE}")

# Load trained classifier
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2),
)

classifier.load_state_dict(checkpoint["classifier_state_dict"])
classifier.to(device)
classifier.eval()

# Load cached validation features and labels
features, labels = torch.load(
    VALIDATION_CACHE,
    map_location=device,
    weights_only=True,
)

features = features.to(device)
labels = labels.to(device).long()

# Predict in batches
all_predictions = []

with torch.no_grad():
    for start in range(0, len(features), 128):
        batch = features[start:start + 128]
        logits = classifier(batch)
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().tolist())

y_true = labels.cpu().numpy()
y_pred = all_predictions

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print("\n========== VGG-16 EVALUATION ==========")
print(f"Validation images: {len(y_true)}")
print(f"Correct predictions: {(y_true == y_pred).sum()}")
print(f"Incorrect predictions: {(y_true != y_pred).sum()}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")

# Classification report
class_names = ["Cat", "Dog"]

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

# Save text report
report_path = OUTPUT_DIR / "vgg16_classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("VGG-16 Validation Evaluation\n")
    file.write(f"Validation accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

display.plot(cmap="Blues", values_format="d")
plt.title("VGG-16 Confusion Matrix")
plt.tight_layout()

cm_path = OUTPUT_DIR / "vgg16_confusion_matrix.png"
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nConfusion matrix saved to: {cm_path}")
print(f"Classification report saved to: {report_path}")

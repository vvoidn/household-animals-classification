
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

models = [
    "Small CNN\n5 epochs",
    "Small CNN\n10 epochs",
    "Small CNN\n20 epochs",
    "VGG-16\ntransfer learning",
]

accuracies = [73.02, 80.84, 87.80, 98.80]

plt.figure(figsize=(9, 5))
bars = plt.bar(models, accuracies)

plt.ylabel("Validation Accuracy (%)")
plt.title("Household Animals Classification: Model Comparison")
plt.ylim(0, 105)
plt.grid(axis="y", linestyle="--", alpha=0.35)

for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(output_dir / "model_comparison.png", dpi=200)
plt.close()

print("Saved comparison graph to outputs/model_comparison.png")


from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)

BASE = Path.cwd()
OUT = BASE / "report.pdf"
OUTPUTS = BASE / "outputs"

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    rightMargin=1.6 * cm,
    leftMargin=1.6 * cm,
    topMargin=1.4 * cm,
    bottomMargin=1.4 * cm,
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=8
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading2"],
    fontSize=12, leading=14, spaceBefore=6, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="BodySmall", parent=styles["BodyText"],
    fontSize=8.5, leading=11, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="Caption", parent=styles["BodyText"],
    fontSize=8, leading=10, alignment=TA_CENTER, spaceAfter=5
))

story = []

# PAGE 1
story.append(Paragraph(
    "Household Animals Classification Using Deep Learning",
    styles["ReportTitle"]
))
story.append(Paragraph(
    "Mini-Project Report | Image Classification: Cats vs. Dogs",
    styles["Normal"]
))
story.append(Spacer(1, 8))

story.append(Paragraph("1. Abstract", styles["Section"]))
story.append(Paragraph(
    "This project implements and compares two deep-learning approaches for "
    "classifying household animal images into cat and dog categories. The first "
    "approach is a custom Small Convolutional Neural Network (CNN), trained "
    "from scratch. The second uses a VGG-16 network pretrained on ImageNet as "
    "a frozen feature extractor, with a newly trained classifier. The recorded "
    "validation accuracies were 87.80% for the 20-epoch Small CNN and 98.80% "
    "for the VGG-16 feature-extraction approach. The results demonstrate the "
    "potential benefit of transfer learning for this image-classification task.",
    styles["BodySmall"]
))

story.append(Paragraph("2. Dataset and Preprocessing", styles["Section"]))
story.append(Paragraph(
    "The dataset contains 25,000 labeled images: 12,500 cats and 12,500 dogs. "
    "The experiments used 20,000 images for training and 5,000 for validation. "
    "The Small CNN resizes images to 128 × 128 pixels. The VGG-16 pipeline "
    "resizes images to 224 × 224 pixels and applies ImageNet channel "
    "normalization. The dataset remains stored locally and is not included "
    "in the project repository.",
    styles["BodySmall"]
))

story.append(Paragraph("3. Methodology", styles["Section"]))
story.append(Paragraph(
    "<b>Small CNN:</b> Four convolutional stages use ReLU activations and "
    "max-pooling, followed by adaptive average pooling, dropout, and a "
    "two-class output layer. Separate runs were recorded at 5, 10, and "
    "20 epochs.<br/><br/>"
    "<b>VGG-16 transfer learning:</b> ImageNet-pretrained VGG-16 convolutional "
    "features are frozen and extracted once for the dataset. The cached "
    "features are used to train a classifier with a 256-unit hidden layer, "
    "ReLU, dropout, and a two-class output layer. This is feature extraction "
    "rather than end-to-end fine-tuning.",
    styles["BodySmall"]
))

story.append(Paragraph("4. Recorded Validation Results", styles["Section"]))
data = [
    ["Model", "Validation accuracy"],
    ["Small CNN — 5 epochs", "73.02%"],
    ["Small CNN — 10 epochs", "80.84%"],
    ["Small CNN — 20 epochs", "87.80%"],
    ["VGG-16 feature extraction", "98.80%"],
]
table = Table(data, colWidths=[10.5 * cm, 5 * cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("PADDING", (0, 0), (-1, -1), 5),
]))
story.append(table)
story.append(Spacer(1, 5))
story.append(Paragraph(
    "The recorded results come from the existing training runs. The validation "
    "splits should be checked for identity before interpreting the comparison "
    "as a controlled head-to-head experiment.",
    styles["BodySmall"]
))

# PAGE 2
story.append(PageBreak())
story.append(Paragraph("5. Evaluation and Visual Results", styles["Section"]))

comparison = OUTPUTS / "model_comparison.png"
small_cm = OUTPUTS / "confusion_matrix.png"
vgg_cm = OUTPUTS / "vgg16_confusion_matrix.png"

def add_image_if_present(path, width, height, caption):
    if path.is_file():
        story.append(Image(str(path), width=width, height=height))
        story.append(Paragraph(caption, styles["Caption"]))
    else:
        story.append(Paragraph(
            f"Image not found: {path.name}", styles["BodySmall"]
        ))

add_image_if_present(
    comparison, 15.5 * cm, 8.1 * cm,
    "Figure 1. Recorded validation accuracy across the four experiments."
)

story.append(Paragraph("Confusion Matrices", styles["Section"]))
add_image_if_present(
    small_cm, 7.2 * cm, 6.0 * cm,
    "Figure 2. Small CNN confusion matrix (20-epoch model)."
)
add_image_if_present(
    vgg_cm, 7.2 * cm, 6.0 * cm,
    "Figure 3. VGG-16 confusion matrix."
)

story.append(Paragraph("6. Discussion and Limitations", styles["Section"]))
story.append(Paragraph(
    "The custom CNN's recorded validation accuracy increased as the training "
    "duration increased. The VGG-16 feature-extraction experiment recorded the "
    "highest validation accuracy. Pretrained features may provide useful "
    "representations for distinguishing visual patterns in animal images. "
    "However, the models use different image resolutions and preprocessing, "
    "and their validation splits may differ. The VGG-16 training history also "
    "showed very high training accuracy and increasing validation loss, so "
    "overfitting should be considered. Validation accuracy alone does not "
    "establish performance on new datasets or guarantee reliable predictions "
    "for every image.",
    styles["BodySmall"]
))

story.append(Paragraph("7. Conclusion", styles["Section"]))
story.append(Paragraph(
    "The project demonstrates a complete image-classification workflow using "
    "a custom CNN and VGG-16 transfer learning, including training, cached "
    "feature extraction, evaluation, confusion matrices, and image prediction. "
    "The recorded VGG-16 result is promising. Future work should compare both "
    "models on an identical validation split, evaluate on an independent test "
    "set, and examine misclassified images.",
    styles["BodySmall"]
))

story.append(Paragraph("Implementation Tools", styles["Section"]))
story.append(Paragraph(
    "Python 3.14.4, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, "
    "Pillow 12.3.0, NumPy 2.5.2, Matplotlib 3.11.2, "
    "scikit-learn 1.9.1, and ReportLab 5.0.1.",
    styles["BodySmall"]
))

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(A4[0] / 2, 0.7 * cm, f"Page {doc.page}")
    canvas.restoreState()

doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"Report created: {OUT}")

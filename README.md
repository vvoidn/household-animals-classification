
# Household Animals Classification Using Deep Learning

## 1. Project Overview

This project classifies household animal images into two categories: **Cat** and **Dog** using deep learning. Two approaches are implemented and evaluated:

1. A custom Small Convolutional Neural Network (CNN).
2. Transfer learning using pretrained VGG-16 features and a newly trained classifier.

The objective is to compare a lightweight CNN trained on the dataset with a model that uses pretrained image features.

## 2. Dataset

- Dataset: Kaggle Dogs vs. Cats
- Total labeled images: 25,000
- Cat images: 12,500
- Dog images: 12,500
- Training split: 20,000 images
- Validation split: 5,000 images

The dataset is stored locally and is not included in this repository.

## 3. Models Implemented

### Small CNN

A custom CNN consisting of convolutional layers, ReLU activations, max-pooling, adaptive average pooling, dropout, and a final two-class classifier.

Separate experiments were conducted for 5, 10, and 20 epochs.

### VGG-16 Transfer Learning

A VGG-16 network pretrained on ImageNet is used as a frozen feature extractor. Its extracted features are cached, and a new classifier is trained to distinguish cats from dogs.

This feature-extraction approach reduces repeated computation during classifier training.

## 4. Validation Results

| Model | Validation Accuracy |
|---|---:|
| Small CNN — 5 epochs | 73.02% |
| Small CNN — 10 epochs | 80.84% |
| Small CNN — 20 epochs | 87.80% |
| VGG-16 transfer learning | 98.80% |

These are the recorded validation results. Confirm that both approaches use the same validation split before treating this as a controlled model comparison.

## 5. Project Structure

```text
animal-classification/
├── prepare_data.py
├── train_cnn.py
├── train_vgg16.py
├── train_vgg16_cached.py
├── evaluate.py
├── evaluate_vgg16_cached.py
├── predict.py
├── predict_vgg16.py
├── plot_model_comparison.py
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   ├── confusion_matrix.png
│   ├── vgg16_confusion_matrix.png
│   ├── vgg16_classification_report.txt
│   ├── model_comparison.png
│   └── vgg16_feature_cache/
└── backups/
```

Model checkpoints are also stored in `outputs/`. The dataset, virtual environment, and large feature-cache files should not be uploaded to GitHub.

## 6. Requirements

- Python
- PyTorch
- torchvision
- Pillow
- NumPy
- Matplotlib
- scikit-learn

## 7. Running the Project

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Predict an image using the Small CNN:

```bash
python predict.py /path/to/image.jpg
```

Predict an image using VGG-16:

```bash
python predict_vgg16.py /path/to/image.jpg
```

Evaluate the Small CNN:

```bash
python evaluate.py
```

Evaluate VGG-16 using cached validation features:

```bash
python evaluate_vgg16_cached.py
```

Generate the model-comparison graph:

```bash
python plot_model_comparison.py
```

Run these commands from the project root directory.

## 8. Conclusion

The recorded results show that VGG-16 transfer learning achieved higher validation accuracy than the custom Small CNN. The project demonstrates the use of convolutional neural networks, pretrained feature extraction, classification metrics, confusion matrices, and model comparison for image classification.

The comparison should be interpreted in light of the validation split, preprocessing, and experimental setup.

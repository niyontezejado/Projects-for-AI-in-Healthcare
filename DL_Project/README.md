# Brain Tumor Classification Using Transfer Learning

> **Course:** Deep Learning and Reinforcement Learning — ISOM 678
>
> **Authors:** Jean De Dieu Niyonteze, Mandingwa Timothy
>
> **Date:** April 24, 2026

---

## Overview

This project classifies brain MRI images into four tumor categories using transfer learning with four pretrained deep learning models. The models are fine-tuned on a medical imaging dataset and evaluated across multiple performance metrics including accuracy, precision, recall, F1-score, and inference time.

---

## Classes

| Label | Description |
|---|---|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `notumor` | No tumor present |
| `pituitary` | Pituitary tumor |

---

## Models

| Model | Input Size | Pretrained On |
|---|---|---|
| VGG16 | 224 × 224 | ImageNet |
| ResNet50 | 224 × 224 | ImageNet |
| Xception | 299 × 299 | ImageNet |
| EfficientNetB2 | 260 × 260 | ImageNet |

---

## Project Structure

```
DL_Project/
├── archive/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── outputs/
│   ├── vgg16_model.keras
│   ├── resnet50_model.keras
│   ├── xception_model.keras
│   ├── efficientnetb2_model.keras
│   ├── *_classification_report.txt
│   ├── *_confusion_matrix.png
│   ├── *_accuracy_loss.png
│   ├── inference_time_comparison.png
│   └── realistic_inference_time_comparison.png
├── brain-tumor-classification.ipynb
├── realtime_inference.py
├── environment.yml
└── README.md
```

---

## Pipeline

### 1. Data Preparation
- Images resized to `224 × 224` and loaded in batches of 32
- Training folder split **80/20** into train and validation sets
- Class weights computed with `sklearn` to handle class imbalance
- Pipeline optimized with `.cache().shuffle().prefetch()`

### 2. Data Augmentation
Applied on-the-fly during training:
- Random horizontal flip
- Random rotation (±10%)
- Random zoom (±10%)

### 3. Model Architecture
Each model follows the same pattern:
- Pretrained backbone loaded with ImageNet weights, top removed
- **Phase 1 — Feature Extraction:** backbone frozen, only custom head trained
- **Phase 2 — Fine-tuning:** last 7 backbone layers unfrozen, retrained at lower learning rate

Custom head added on top of each backbone:
```
GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(4, Softmax)
```

### 4. Training Strategy
| Setting | Phase 1 | Phase 2 |
|---|---|---|
| Epochs | up to 25 | up to 10 |
| Learning Rate | `1e-4` | `1e-5` |
| Early Stopping | patience=5 | patience=5 |
| ReduceLROnPlateau | factor=0.2, patience=3 | factor=0.2, patience=3 |
| Class Weights | yes | yes |

### 5. Evaluation
- Test set accuracy and loss
- Confusion matrix (heatmap)
- Classification report: precision, recall, F1-score per class
- Cross-model comparison: macro-average metrics bar chart
- Inference time benchmark (dummy tensors + real MRI images)

---

## Setup

### 0. Clone and Access the Project Repository

Clone the repository
```bash
git clone https://github.com/niyontezejado/Projects-for-AI-in-Healthcare
```
Access the repository
```bash
cd Projects-for-AI-in-Healthcare\DL_Project
```

### 1. Create the Conda Environment

A ready-to-use environment file is provided. Run this once to create the environment:

```bash
conda env create -f environment.yml
```

This creates a conda environment named **`DeepLearning`** with all required dependencies



### 2. Activate the Environment

```bash
conda activate DeepLearning
```

---

## Running the Notebook

Open and run `brain-tumor-classification.ipynb` top to bottom.
Trained models and all outputs are saved to the `outputs/` folder automatically.

---

## Real-Time Inference Script

`realtime_inference.py` loads a saved model and runs predictions on real MRI test images, displaying an annotated result grid.

### Usage

```bash
# Default: VGG16, 5 images per class
python realtime_inference.py

# Choose model
python realtime_inference.py --model EfficientNetB2

# Choose model and number of images per class
python realtime_inference.py --model ResNet50 --samples-per-class 10
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `VGG16` | Model to use: `VGG16`, `ResNet50`, `Xception`, `EfficientNetB2` |
| `--samples-per-class` | `5` | Number of test images per class (total = N × 4) |

### Output Display

Each image is shown alongside an annotation panel containing:
- **Correct / Wrong** verdict badge (green / red)
- Ground Truth (GT) class label
- Predicted class label
- Per-image inference time (ms)

Press **Ctrl+C** at any time to stop predictions early — results collected so far will still be displayed.

---

## Outputs

### Saved Models
| File | Description |
|---|---|
| `vgg16_model.keras` | Trained VGG16 model |
| `resnet50_model.keras` | Trained ResNet50 model |
| `xception_model.keras` | Trained Xception model |
| `efficientnetb2_model.keras` | Trained EfficientNetB2 model |

### Classification Reports
| File | Description |
|---|---|
| `vgg16_classification_report.txt` | Per-class precision, recall, F1 — VGG16 |
| `resnet50_classification_report.txt` | Per-class precision, recall, F1 — ResNet50 |
| `xception_classification_report.txt` | Per-class precision, recall, F1 — Xception |
| `efficientnetb2_classification_report.txt` | Per-class precision, recall, F1 — EfficientNetB2 |

### Training Curves
| File | Description |
|---|---|
| `vgg16_training_curves.png` | Accuracy & loss curves — VGG16 |
| `resnet50_accuracy_loss.png` | Accuracy & loss curves — ResNet50 |
| `xception_accuracy_loss.png` | Accuracy & loss curves — Xception |
| `efficientnetb2_accuracy_loss.png` | Accuracy & loss curves — EfficientNetB2 |

### Confusion Matrices
| File | Description |
|---|---|
| `vgg16_confusion_matrix.png` | Confusion matrix — VGG16 |
| `resnet50_confusion_matrix.png` | Confusion matrix — ResNet50 |
| `xception_confusion_matrix.png` | Confusion matrix — Xception |
| `efficientnetb2_confusion_matrix.png` | Confusion matrix — EfficientNetB2 |

### Prediction Grids
| File | Description |
|---|---|
| `vgg16_prediction_grid.png` | Annotated test predictions — VGG16 |
| `resnet_50_prediction_grid.png` | Annotated test predictions — ResNet50 |
| `xception_prediction_grid.png` | Annotated test predictions — Xception |
| `efficientnet_b2_prediction_grid.png` | Annotated test predictions — EfficientNetB2 |

### Dataset & Comparison Plots
| File | Description |
|---|---|
| `class_distribution.png` | Image count per class in training set |
| `class_distribution_augmented.png` | Image count per class after augmentation |
| `overall_metrics_comparison.png` | Macro-average accuracy, precision, recall, F1 across all models |
| `metrics_facet_by_class.png` | Per-class P/R/F1 comparison across all models |
| `inference_time_comparison.png` | Dummy-tensor inference time benchmark |
| `realistic_inference_time_comparison.png` | Real MRI image inference time benchmark |
| `realtime_inference_efficientnetb2.png` | Real-time inference annotated grid from inference script |

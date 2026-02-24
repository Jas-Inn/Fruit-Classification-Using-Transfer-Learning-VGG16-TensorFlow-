# Fruit-Classification-Using-Transfer-Learning-VGG16-TensorFlow-
A deep learning image classifier that identifies different fruit categories using transfer learning with the pre-trained VGG16 model. The project covers the full ML pipeline — from data loading and augmentation, to model training, fine-tuning, evaluation, and visual prediction.

# Project Overview
Training image classifiers from scratch requires massive datasets and compute. Transfer learning sidesteps this by reusing a model already trained on millions of images (ImageNet) and adapting it to a new task with far less data and time.
In this project, VGG16 serves as the feature extractor. Its convolutional layers are frozen initially, and custom classification layers are stacked on top. A second fine-tuning phase then unfreezes the last few VGG16 layers, allowing the model to adapt lower-level features to fruit-specific patterns.


Training image classifiers from scratch requires massive datasets and compute. Transfer learning sidesteps this by reusing a model already trained on millions of images (ImageNet) and adapting it to a new task with far less data and time.

In this project, VGG16 serves as the feature extractor. Its convolutional layers are frozen initially, and custom classification layers are stacked on top. A second fine-tuning phase then unfreezes the last few VGG16 layers, allowing the model to adapt lower-level features to fruit-specific patterns.

---

# Repository Structure

```
fruit-classification-vgg16/
├── fruit_classification_vgg16.ipynb   # Main notebook — full pipeline
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignores data, model weights, etc.
├── README.md                          # Instructions 
└── outputs/                           # Saved plots and model artifacts (optional)
```

---

# Dataset

This project uses the **Fruits-360 (Original Size)** dataset — a publicly available benchmark dataset of fruit images.

- **24 fruit classes** (including apple varieties, pear, cucumber, and more)
- **~12,500 images** split across Training, Validation, and Test sets:
  - Training: 6,231 images
  - Validation: 3,114 images
  - Test: 3,110 images
- Each image resized to **64×64 pixels** during preprocessing

The dataset is downloaded automatically by the notebook. See [`data/README.md`](data/README.md) for manual download instructions.

## Manual Download

If automatic download fails, you can download the dataset manually:

1. Go to the [Fruits-360 dataset on Kaggle](https://www.kaggle.com/datasets/moltean/fruits)
2. Download and extract it into the project root
3. Ensure the directory structure looks like this:

```
fruits-360-original-size/
└── fruits-360-original-size/
    ├── Training/
    │   ├── apple_braeburn_1/
    │   ├── pear_1/
    │   ├── cucumber_3/
    │   └── ... (24 class folders)
    ├── Validation/
    │   └── ... (same 24 class folders)
    └── Test/
        └── ... (same 24 class folders)
```

## Dataset Stats

| Split | Images | Classes |
|---|---|---|
| Training | 6,231 | 24 |
| Validation | 3,114 | 24 |
| Test | 3,110 | 24 |
---

# Model Architecture

| Component | Details |
|---|---|
| **Base model** | VGG16 (ImageNet weights, `include_top=False`) |
| **Pooling** | GlobalAveragePooling2D |
| **Custom head** | Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Dense(24, Softmax) |
| **Loss** | Categorical Crossentropy |
| **Optimizer** | Adam |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau |

---

# Training Pipeline

**Phase 1 — Feature Extraction (5 epochs)**
All VGG16 layers are frozen. Only the custom head is trained.

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 26.8% | 21.0% |
| 3 | 58.6% | 46.5% |
| 5 | 69.1% | 65.3% |

**Phase 2 — Fine-Tuning (5 epochs)**
Last 5 VGG16 layers unfrozen. Model re-compiled with a low learning rate (1e-5).

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 73.8% | 64.5% |
| 3 | 84.6% | 75.3% |
| 5 | 85.4% | 82.5% |

Fine-tuning brought a significant boost — from ~65% to ~85% validation accuracy.

---

# Data Augmentation

To improve generalization, the training generator applies the following augmentations at runtime:

- Random rotation (±20°)
- Width and height shifts (±10–20%)
- Shear and zoom (±20%)
- Horizontal flips

Validation and test sets are only rescaled (no augmentation), ensuring unbiased evaluation.

---

# Results

The notebook generates:
- **Accuracy and loss curves** across training epochs (both phases)
- **Per-image prediction visualizations** comparing actual vs. predicted class labels

Misclassifications were most common between visually similar classes (e.g., different apple varieties), which is expected behavior given the 64×64 input resolution.

---

# How to Run

## 1. Clone the repo
```bash
git clone https://github.com/your-username/fruit-classification-vgg16.git
cd fruit-classification-vgg16
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Launch the notebook
```bash
jupyter notebook fruit_classification_vgg16.ipynb
```

The dataset (~500MB) will be downloaded automatically when you run the notebook. Make sure you have a stable internet connection.

> **Note:** Training is resource-intensive. A GPU is recommended. On CPU, expect ~1–2 minutes per epoch.

---

# Tech Stack

- Python 3.12
- TensorFlow 2.16 / Keras 3
- NumPy, Matplotlib, Scikit-learn
- Fruits-360 dataset

---

# Key Concepts Demonstrated

- Transfer learning with frozen and fine-tuned layers
- Data augmentation with `ImageDataGenerator`
- Callbacks: `EarlyStopping` and `ReduceLROnPlateau`
- Multi-class image classification (24 classes)
- Model evaluation on held-out test data
- Prediction visualization with class label mapping

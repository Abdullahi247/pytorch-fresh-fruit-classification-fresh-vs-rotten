# pytorch-fresh-fruit-classification-fresh-vs-rotten
This project is based on  training and validating computer vision models for fruit quality detection, automated sorting, and AI-based freshness monitoring

# 🍎 Fresh vs Rotten Fruit Classification (Multi-Task Learning)

This project implements a **multi-task deep learning system** that simultaneously:

* **Identifies the type of fruit** (e.g. Apple, Banana, Strawberry)
* **Determines its freshness state** (Fresh or Rotten)

using a **shared CNN backbone with task-specific heads**. The model is trained using **PyTorch** and leverages **transfer learning with ResNet50**.

---

## 📌 Project Motivation

In real-world food quality inspection systems (e.g. smart agriculture, retail automation, food safety), it is often insufficient to only classify *what* an object is. We also need to know *its condition*.

Rather than training two separate models, this project adopts **multi-task learning**, allowing:

* Shared visual understanding of fruits
* Better generalization
* Reduced model size and training cost

---

## 🧠 Model Architecture

### 🔹 Backbone

* **ResNet50 (ImageNet pretrained)**
* Early layers frozen for stability
* Last block (layer4) fine-tuned

### 🔹 Shared Representation

```text
Image → ResNet50 → Shared MLP (512-dim)
```

### 🔹 Task Heads

* **Fruit Classification Head** → Multi-class (Apple, Banana, Strawberry, …)
* **Freshness Classification Head** → Binary (Fresh / Rotten)

```text
Shared Features
   ├── Fruit Head → CrossEntropyLoss
   └── Freshness Head → CrossEntropyLoss
```

Total loss is computed as:

```math
L = L_fruit + L_freshness
```

---

## 🗂 Dataset Structure

Dataset is automatically downloaded from Kaggle and organized as:

```text
Fruit Freshness Dataset/
├── Apple/
│   ├── Fresh/
│   └── Rotten/
├── Banana/
│   ├── Fresh/
│   └── Rotten/
└── Strawberry/
    ├── Fresh/
    └── Rotten/
```

Each image path encodes both labels:

* **Fruit class** → folder name
* **Freshness state** → sub-folder name

---

## 🔄 Data Processing & Augmentation

### Training Augmentations

Applied *randomly per epoch* to improve robustness:

* Resize (224 × 224)
* Random Horizontal & Vertical Flip
* Random Rotation (±45°)
* Gaussian Blur
* Sharpness Adjustment
* ImageNet Normalization

> ⚠️ Validation data uses **no augmentation**, only resizing and normalization.

---

## 🧪 Data Splitting & Leakage Prevention

To prevent **data leakage**:

* Image filenames are deduplicated using their stems
* Train/Validation split is performed **after deduplication**
* Overlap between splits is explicitly checked and enforced to be zero

This ensures fair evaluation and realistic performance metrics.

---

## 📊 Training Setup

* **Optimizer:** Adam
* **Learning Rate:** 1e-4
* **Weight Decay:** 1e-3
* **Loss Functions:**

  * Fruit → CrossEntropyLoss
  * Freshness → CrossEntropyLoss

Metrics tracked **independently per task**:

* Training & Validation Loss (Fruit / Freshness)
* Training & Validation Accuracy (Fruit / Freshness)

---

## 📈 Results & Monitoring

The training loop records:

* 📉 Separate loss curves for fruit and freshness
* 📈 Separate accuracy curves for fruit and freshness

This makes it easy to:

* Detect overfitting
* Identify task imbalance
* Monitor negative transfer between tasks

---

## 🌐 Inference on Internet Images

The project includes a utility to:

* Download a random image from the internet
* Apply inference-time preprocessing
* Predict:

  * Fruit type
  * Freshness state

Predictions are automatically decoded from class indices to human-readable labels.

---

## 🛠 Tech Stack

* **Python**
* **PyTorch & TorchVision**
* **ResNet50 (Transfer Learning)**
* **OpenCV / PIL**
* **Matplotlib**
* **Scikit-learn**

---

## 🚀 Future Improvements

* Task-weighted loss balancing
* Gradual unfreezing of backbone layers
* Confidence-based predictions
* Real-time webcam inference
* Deployment via FastAPI or Streamlit

---

## ✅ Key Takeaways

* Multi-task learning improves efficiency and representation sharing
* Careful loss & metric separation is critical
* Preventing data leakage is essential for trustworthy results

---

## 👨‍💻 Author @ Abdullah Yusuf

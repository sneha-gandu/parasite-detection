# 🔬 Deep Learning-Based Veterinary Parasite Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?logo=tensorflow&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> Automated classification of veterinary parasites from microscopic images using **Transfer Learning (InceptionV3)**. Achieves **94% accuracy** — approximately 40% improvement over manual microscopic examination.

---

## 🧬 Problem Statement

Identifying parasites from microscope slides is time-consuming, requires expert veterinary knowledge, and is prone to human error. This project automates classification of 5 common veterinary parasites using deep learning, enabling faster and more consistent diagnosis in veterinary clinics.

---

## 📊 Dataset

| Detail | Value |
|---|---|
| Classes | Ascaris, Hookworm, Pinworm, Tapeworm, Toxocara |
| Image format | JPG / PNG (microscopic slides) |
| Input size | 299 × 299 px (InceptionV3 standard) |
| Split | 80% train / 20% validation + held-out test set |

Organise your dataset as:
```
data/
├── train/
│   ├── Ascaris/
│   ├── Hookworm/
│   ├── Pinworm/
│   ├── Tapeworm/
│   └── Toxocara/
└── test/
    ├── Ascaris/
    └── ...
```

---

## 🏗️ Model Architecture

```
InceptionV3 (ImageNet pretrained, frozen)
        ↓
GlobalAveragePooling2D
        ↓
Dense(512) + ReLU + Dropout(0.4)
        ↓
Dense(256) + ReLU + Dropout(0.3)
        ↓
Dense(5)   + Softmax   ← parasite classes
```

**Training strategy:**
1. **Phase 1** — Freeze InceptionV3 base; train custom head only (10 epochs, lr=1e-4)
2. **Phase 2** — Unfreeze last 50 layers; fine-tune end-to-end (15 epochs, lr=1e-5)

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/parasite-detection.git
cd parasite-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare dataset (see structure above)

# 4. Train the model
python train.py

# 5. Predict on a new image
python predict.py --image path/to/microscope_image.jpg
```

---

## 📈 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **94.0%** |
| Precision (macro avg) | 93.5% |
| Recall (macro avg) | 93.2% |
| F1-Score (macro avg) | 93.3% |
| Improvement over manual | ~40% |

### Confusion Matrix
![Confusion Matrix](outputs/plots/confusion_matrix.png)

### Training Curves
![Training Curves](outputs/plots/training_curves.png)

---

## 🗂️ Project Structure

```
parasite-detection/
├── data/                        # Dataset (train/ + test/ folders)
├── model/
│   └── parasite_model.h5        # Saved trained model
├── outputs/
│   └── plots/
│       ├── confusion_matrix.png
│       └── training_curves.png
├── train.py                     # Full training pipeline
├── predict.py                   # Single-image inference
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.x / Keras
- **Model**: InceptionV3 (Transfer Learning)
- **Image Processing**: OpenCV, PIL
- **Evaluation**: scikit-learn (classification report, confusion matrix)
- **Visualisation**: Matplotlib, Seaborn

---

## 🔮 Future Improvements

- [ ] Expand to 10+ parasite classes
- [ ] Build a Streamlit web app for clinic use
- [ ] Add Grad-CAM visualisation (show which regions model focuses on)
- [ ] Deploy as REST API with FastAPI

---

## 📄 License

MIT License — feel free to use and build on this project.

---

*Developed as part of B.Tech project — Sri Venkateswara Engineering College*

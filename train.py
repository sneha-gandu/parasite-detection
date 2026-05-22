"""
Deep Learning-Based Veterinary Parasite Detection
Author: [Your Name]
Model: InceptionV3 (Transfer Learning) | Framework: TensorFlow/Keras
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR      = "data/"          # subfolders = class names
IMG_SIZE      = (299, 299)
BATCH_SIZE    = 32
EPOCHS        = 25
LEARNING_RATE = 1e-4
MODEL_SAVE    = "model/parasite_model.h5"

CLASSES = ["Ascaris", "Hookworm", "Pinworm", "Tapeworm", "Toxocara"]
NUM_CLASSES = len(CLASSES)

# ─────────────────────────────────────────────
# 1. DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
print("=" * 55)
print("  VETERINARY PARASITE DETECTION — TRAINING PIPELINE")
print("=" * 55)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

print("\n[1/5] Loading training data...")
train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"  ✓ Train samples   : {train_gen.samples}")
print(f"  ✓ Val samples     : {val_gen.samples}")
print(f"  ✓ Test samples    : {test_gen.samples}")
print(f"  ✓ Classes detected: {list(train_gen.class_indices.keys())}")

# ─────────────────────────────────────────────
# 2. BUILD MODEL (InceptionV3 + Custom Head)
# ─────────────────────────────────────────────
print("\n[2/5] Building InceptionV3 model (Transfer Learning)...")

base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)

# Freeze all base layers initially
base_model.trainable = False
print(f"  ✓ InceptionV3 base loaded — {len(base_model.layers)} layers (frozen)")

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"  ✓ Model compiled  — Total params: {model.count_params():,}")

# ─────────────────────────────────────────────
# 3. TRAIN — PHASE 1 (head only)
# ─────────────────────────────────────────────
print("\n[3/5] Training Phase 1 — Custom head only (10 epochs)...")

os.makedirs("model", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_SAVE, save_best_only=True, verbose=0),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# Fine-tune: unfreeze last 50 layers
print("\n  Fine-tuning — unfreezing last 50 layers of InceptionV3...")
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n[3/5] Training Phase 2 — Fine-tuning (15 epochs)...")
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 4. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print("\n[4/5] Evaluating on test set...")

test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\n  ┌─────────────────────────────────┐")
print(f"  │  Test Accuracy : {test_acc*100:.2f}%          │")
print(f"  │  Test Loss     : {test_loss:.4f}           │")
print(f"  └─────────────────────────────────┘")

# Predictions
test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes
class_names = list(test_gen.class_indices.keys())

print("\n  Classification Report:")
print("  " + "-" * 55)
report = classification_report(y_true, y_pred, target_names=class_names)
for line in report.split("\n"):
    print("  " + line)

# ─────────────────────────────────────────────
# 5. SAVE PLOTS
# ─────────────────────────────────────────────
print("\n[5/5] Saving plots...")

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix — Parasite Classification", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig("outputs/plots/confusion_matrix.png", dpi=150)
plt.close()
print("  ✓ Saved: outputs/plots/confusion_matrix.png")

# --- Training Accuracy/Loss Curves ---
# Combine both phases
all_acc = history1.history["accuracy"] + history2.history["accuracy"]
all_val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
all_loss = history1.history["loss"] + history2.history["loss"]
all_val_loss = history1.history["val_loss"] + history2.history["val_loss"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(all_acc,     label="Train Accuracy",  color="#2196F3", linewidth=2)
ax1.plot(all_val_acc, label="Val Accuracy",    color="#F44336", linewidth=2, linestyle="--")
ax1.axvline(x=len(history1.history["accuracy"]) - 1,
            color="gray", linestyle=":", label="Fine-tune start")
ax1.set_title("Model Accuracy", fontsize=13)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(all_loss,     label="Train Loss",  color="#2196F3", linewidth=2)
ax2.plot(all_val_loss, label="Val Loss",    color="#F44336", linewidth=2, linestyle="--")
ax2.axvline(x=len(history1.history["loss"]) - 1,
            color="gray", linestyle=":", label="Fine-tune start")
ax2.set_title("Model Loss", fontsize=13)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle("InceptionV3 Parasite Detector — Training History", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("outputs/plots/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved: outputs/plots/training_curves.png")

model.save(MODEL_SAVE)
print(f"\n  ✓ Model saved: {MODEL_SAVE}")
print("\n" + "=" * 55)
print("  TRAINING COMPLETE")
print(f"  Final Test Accuracy: {test_acc*100:.2f}%")
print("=" * 55)

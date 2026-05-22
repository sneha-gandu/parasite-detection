"""
predict.py — Predict parasite class from a single image
Usage: python predict.py --image path/to/image.jpg
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_NAMES = ["Ascaris", "Hookworm", "Pinworm", "Tapeworm", "Toxocara"]
MODEL_PATH  = "model/parasite_model.h5"
IMG_SIZE    = (299, 299)

def predict(img_path: str):
    model = load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    # ── Print results ──────────────────────────────────────
    print("\n" + "=" * 45)
    print("  PARASITE DETECTION RESULT")
    print("=" * 45)
    print(f"  Image     : {img_path}")
    print(f"  Prediction: {pred_class}")
    print(f"  Confidence: {confidence:.2f}%")
    print("-" * 45)
    print("  All class probabilities:")
    for cls, prob in zip(CLASS_NAMES, probs):
        bar = "█" * int(prob * 30)
        marker = " ◄" if cls == pred_class else ""
        print(f"    {cls:<12} {prob*100:5.1f}%  {bar}{marker}")
    print("=" * 45)

    # ── Plot ───────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.imshow(plt.imread(img_path))
    ax1.axis("off")
    ax1.set_title(f"Input: {img_path.split('/')[-1]}", fontsize=11)

    colors = ["#F44336" if c == pred_class else "#90CAF9" for c in CLASS_NAMES]
    bars = ax2.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor="white", height=0.6)
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("Confidence (%)", fontsize=11)
    ax2.set_title("Class Probabilities", fontsize=12)
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                 f"{prob*100:.1f}%", va="center", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    red_patch = mpatches.Patch(color="#F44336", label=f"Predicted: {pred_class} ({confidence:.1f}%)")
    ax2.legend(handles=[red_patch], loc="lower right", fontsize=10)

    plt.suptitle("Veterinary Parasite Detection — InceptionV3", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/prediction_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  ✓ Plot saved: outputs/prediction_result.png")

    return pred_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict parasite class from microscope image")
    parser.add_argument("--image", required=True, help="Path to input image (JPG/PNG)")
    args = parser.parse_args()
    predict(args.image)

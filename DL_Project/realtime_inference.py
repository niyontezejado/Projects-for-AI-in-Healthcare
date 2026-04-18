import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import tensorflow as tf
from tensorflow.keras.applications import vgg16, resnet50, xception

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Real-time brain tumor inference: predicts MRI images and displays annotated results."
)
parser.add_argument(
    "--model",
    choices=["VGG16", "ResNet50", "Xception", "EfficientNetB2"],
    default="VGG16",
    help="Model to use for inference (default: VGG16)"
)
parser.add_argument(
    "--samples-per-class",
    type=int,
    default=5,
    metavar="N",
    help="Number of images per class to predict (default: 5, total = N x 4 classes)"
)
args = parser.parse_args()
SELECTED_MODEL    = args.model
SAMPLES_PER_CLASS = args.samples_per_class

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TEST_DIR   = os.path.join(BASE_DIR, "archive", "Testing")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load selected model ───────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "VGG16":          ("vgg16_model.keras",         {"preprocess_input": vgg16.preprocess_input}),
    "ResNet50":       ("resnet50_model.keras",       {"preprocess_input": resnet50.preprocess_input}),
    "Xception":       ("xception_model.keras",       {"preprocess_input": xception.preprocess_input}),
    "EfficientNetB2": ("efficientnetb2_model.keras", {}),
}

fname, custom_obj = MODEL_CONFIGS[SELECTED_MODEL]
print(f"Loading {SELECTED_MODEL}...")
model = tf.keras.models.load_model(
    os.path.join(OUTPUT_DIR, fname),
    custom_objects=custom_obj or None
)
print(f"{SELECTED_MODEL} loaded.\n")

# ── Load images and run predictions ──────────────────────────────────────────
class_folders = sorted(os.listdir(TEST_DIR))

results = []  # {image, gt, predicted, inference_ms, correct}

print("Running predictions... Press Ctrl+C to stop at any time.\n")
try:
    for cls_idx, cls_name in enumerate(class_folders):
        cls_dir   = os.path.join(TEST_DIR, cls_name)
        img_files = sorted(os.listdir(cls_dir))[:SAMPLES_PER_CLASS]

        for img_fname in img_files:
            img_path  = os.path.join(cls_dir, img_fname)
            img       = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_arr   = tf.keras.utils.img_to_array(img)
            img_input = np.expand_dims(img_arr, axis=0).astype(np.float32)

            start      = time.time()
            probs      = model.predict(img_input, verbose=0)
            elapsed_ms = (time.time() - start) * 1000

            pred_idx  = int(np.argmax(probs, axis=1)[0])
            pred_name = class_folders[pred_idx]
            correct   = pred_idx == cls_idx

            results.append({
                "image":        img_arr.astype(np.uint8),
                "gt":           cls_name,
                "predicted":    pred_name,
                "inference_ms": elapsed_ms,
                "correct":      correct,
            })
            status = "✓" if correct else "✗"
            print(f"  [{status}] GT: {cls_name:<12} Pred: {pred_name:<12} {elapsed_ms:.1f} ms")

except KeyboardInterrupt:
    print("\n\nStopped by user. Showing results collected so far...\n")

if not results:
    print("No predictions made. Exiting.")
    exit(0)

n_correct = sum(r["correct"] for r in results)
print(f"\nTotal: {len(results)} images  |  Correct: {n_correct}  |  Accuracy: {100*n_correct/len(results):.1f}%\n")

# ── Display annotated grid (image left | annotations right) ──────────────────
n_cols = SAMPLES_PER_CLASS
n_rows = len(class_folders)

fig = plt.figure(figsize=(n_cols * 7 + 2.5, n_rows * 3.5 + 1))

# Title inside the figure with enough top margin
fig.text(
    0.5, 0.97,
    f"Real-Time Inference  —  {SELECTED_MODEL}          "
    f"Accuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.1f}%)",
    fontsize=14, fontweight="bold", ha="center", va="top"
)

# 2 sub-columns per image: [image(5x) | annotation(1.8x)], no gap between them
gs = GridSpec(
    n_rows, n_cols * 2,
    width_ratios=[5, 1.8] * n_cols,
    figure=fig,
    left=0.02, right=0.98,
    top=0.92, bottom=0.07,
    wspace=0.0, hspace=0.1
)

for i, r in enumerate(results):
    row     = i // n_cols
    col     = i  % n_cols
    color   = "green" if r["correct"] else "red"
    verdict = "Correct" if r["correct"] else "Wrong"

    # ── Image (fills its cell completely) ──
    ax_img = fig.add_subplot(gs[row, col * 2])
    ax_img.imshow(r["image"])
    ax_img.axis("off")

    # ── Annotation panel ──
    ax_txt = fig.add_subplot(gs[row, col * 2 + 1])
    ax_txt.set_facecolor("#f2f2f2")
    for spine in ax_txt.spines.values():
        spine.set_visible(False)
    ax_txt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ax_txt.text(0.5, 0.97, verdict,
                transform=ax_txt.transAxes,
                fontsize=12, ha="center", va="top",
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, linewidth=0))

    ax_txt.text(0.5, 0.72, "GT",
                transform=ax_txt.transAxes,
                fontsize=9, ha="center", color="gray", fontstyle="italic")
    ax_txt.text(0.5, 0.61, r["gt"],
                transform=ax_txt.transAxes,
                fontsize=11, ha="center", fontweight="bold", color="black")

    ax_txt.text(0.5, 0.46, "Predicted",
                transform=ax_txt.transAxes,
                fontsize=9, ha="center", color="gray", fontstyle="italic")
    ax_txt.text(0.5, 0.35, r["predicted"],
                transform=ax_txt.transAxes,
                fontsize=11, ha="center", fontweight="bold", color=color)

    ax_txt.text(0.5, 0.20, "Inference",
                transform=ax_txt.transAxes,
                fontsize=9, ha="center", color="gray", fontstyle="italic")
    ax_txt.text(0.5, 0.09, f"{r['inference_ms']:.1f} ms",
                transform=ax_txt.transAxes,
                fontsize=11, ha="center", fontweight="bold", color="dimgray")

# Hide unused cells if stopped early
for j in range(len(results), n_rows * n_cols):
    fig.add_subplot(gs[j // n_cols, (j % n_cols) * 2]).axis("off")
    fig.add_subplot(gs[j // n_cols, (j % n_cols) * 2 + 1]).axis("off")

# Legend
correct_patch = mpatches.Patch(color="green", label="Correct prediction")
wrong_patch   = mpatches.Patch(color="red",   label="Wrong prediction")
fig.legend(
    handles=[correct_patch, wrong_patch],
    loc="lower center", ncol=2, fontsize=10,
    bbox_to_anchor=(0.5, 0.0), frameon=True
)

# Center window at 80% of screen size
try:
    manager = plt.get_current_fig_manager()
    sw = manager.window.winfo_screenwidth()
    sh = manager.window.winfo_screenheight()
    win_w = int(sw * 0.80)
    win_h = int(sh * 0.80)
    x = (sw - win_w) // 2
    y = (sh - win_h) // 2
    manager.window.geometry(f"{win_w}x{win_h}+{x}+{y}")
except Exception:
    pass

out_path = os.path.join(OUTPUT_DIR, f"realtime_inference_{SELECTED_MODEL.lower()}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")

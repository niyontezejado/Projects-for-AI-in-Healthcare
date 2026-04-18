#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =========================
# Paths & basic config
# =========================
from pathlib import Path

# ---- EDIT THESE ----
TRAIN_IMG_DIR = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\train\img")
TRAIN_GT_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\train\vessel")
TEST_IMG_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\test\img")
TEST_GT_DIR   = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\test\vessel")

# Output root (all logs, checkpoints, predictions go here)
OUT_ROOT = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\train\rite_segformer_kfold_outputs")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Model/Training params
ENCODER_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"  # HF base checkpoint, will be fine-tuned
IMG_SIZE = (512, 512)      # (H, W)
NUM_LABELS = 2             # background, vessel
MAX_EPOCHS = 100          # adjust as needed
BATCH_SIZE = 4             # adjust to your GPU
LR = 1e-4
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
SEED = 42
NUM_WORKERS = 0            # set >0 if your env supports
FOLDS = 5                  # KFold
DEVICE_FALLBACK = "cuda"   # "cuda" or "cpu" default preference


# In[2]:


# If needed, install libraries (comment out if your env already has them)
import sys, subprocess, pkgutil

def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

needed = []
for p in ["torch", "torchvision", "transformers", "timm", "albumentations", "opencv-python", "matplotlib", "pandas", "numpy", "scikit-learn"]:
    if pkgutil.find_loader(p) is None:
        needed.append(p)

if needed:
    print("Installing:", needed)
    _pip_install(needed)
else:
    print("All required packages already installed.")


# In[3]:


# ====== 1) PATHS (update if needed) ======
from pathlib import Path

TRAIN_IMG_DIR = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\train\img")
TRAIN_GT_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\train\vessel")
TEST_IMG_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\test\img")
TEST_GT_DIR   = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\test\vessel")

assert TRAIN_IMG_DIR.exists(), TRAIN_IMG_DIR
assert TRAIN_GT_DIR.exists(),  TRAIN_GT_DIR
assert TEST_IMG_DIR.exists(),  TEST_IMG_DIR
assert TEST_GT_DIR.exists(),   TEST_GT_DIR

# ====== 2) Robust pairing helpers ======
IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"]

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def find_mask_for_image(img_path: Path, gt_dir: Path) -> Path | None:
    """
    Try multiple naming patterns used across DRIVE/STARE/RITE/LES-AV.
    Examples:
      IDRiD_01.png -> IDRiD_01_vessel.png
      21_training.png -> 21_manual1.gif
      foo.png -> foo_mask.png
    """
    stem = img_path.stem
    candidates = []

    # exact same stem
    for ext in IMG_EXTS:
        candidates.append(gt_dir / f"{stem}{ext}")

    # common suffixes
    suffixes = ["_vessel", "_vessels", "_manual1", "_mask", "-vessel", "-vessels", "-mask"]
    for suf in suffixes:
        for ext in IMG_EXTS:
            candidates.append(gt_dir / f"{stem}{suf}{ext}")

    # try case-variants
    stems_to_try = {stem, stem.lower(), stem.upper()}
    for st in stems_to_try:
        for suf in suffixes:
            for ext in IMG_EXTS:
                candidates.append(gt_dir / f"{st}{suf}{ext}")

    for c in candidates:
        if c.exists():
            return c
    return None

def list_pairs(img_dir: Path, gt_dir: Path, require_mask: bool = True):
    imgs = sorted([p for p in img_dir.rglob("*") if is_img(p)])
    if not imgs:
        print(f"[WARN] No images found under {img_dir}")
        return [], []

    pairs_img, pairs_gt, missing = [], [], 0
    for ip in imgs:
        gp = find_mask_for_image(ip, gt_dir)
        if gp is None:
            missing += 1
            if require_mask:
                continue
        pairs_img.append(ip)
        pairs_gt.append(gp)

    if missing:
        print(f"[INFO] {missing} masks not found in {gt_dir} (kept={not require_mask}).")

    return pairs_img, pairs_gt

# ====== 3) Build train/test lists ======
train_imgs, train_gts = list_pairs(TRAIN_IMG_DIR, TRAIN_GT_DIR, require_mask=True)
test_imgs,  test_gts  = list_pairs(TEST_IMG_DIR,  TEST_GT_DIR,  require_mask=True)

print(f"Train pairs: {len(train_imgs)} | Test pairs: {len(test_imgs)}")
print("Sample paths:")
for i in range(min(3, len(train_imgs))):
    print("  ", train_imgs[i].name, "->", Path(train_gts[i]).name if train_gts[i] else None)

# Safety checks (now they should pass)
assert len(train_imgs) > 0, f"No train images (after pairing) in {TRAIN_IMG_DIR}"



# In[4]:


# ======================
# Robust file pairing (train + test)
# ======================
IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"]

def is_img(p: Path):
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def find_mask_for_image(img_path: Path, gt_dir: Path) -> Path | None:
    """Try matching vessel/mask filenames with common suffixes."""
    stem = img_path.stem
    suffixes = ["", "_vessel", "_vessels", "_manual1", "_mask", "-vessel", "-mask"]
    for suf in suffixes:
        for ext in IMG_EXTS:
            cand = gt_dir / f"{stem}{suf}{ext}"
            if cand.exists():
                return cand
    return None

def list_pairs(img_dir: Path, gt_dir: Path | None = None, require_mask: bool = True):
    imgs = sorted([p for p in img_dir.rglob("*") if is_img(p)])
    if not imgs:
        print(f"[WARN] No images found under {img_dir}")
        return [], []
    pairs_img, pairs_gt = [], []
    for ip in imgs:
        gp = find_mask_for_image(ip, gt_dir) if gt_dir and gt_dir.exists() else None
        if gp is None and require_mask:
            continue  # skip if training and no GT
        pairs_img.append(ip)
        pairs_gt.append(gp)
    return pairs_img, pairs_gt


# ======================
# Build train/test lists
# ======================
TRAIN_IMG_DIR = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\train\img")
TRAIN_GT_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\train\vessel")
TEST_IMG_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\test\img")
TEST_GT_DIR   = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\test\vessel")  # may not exist

train_imgs, train_gts = list_pairs(TRAIN_IMG_DIR, TRAIN_GT_DIR, require_mask=True)
test_imgs,  test_gts  = list_pairs(TEST_IMG_DIR,  TEST_GT_DIR,  require_mask=False)

print(f"✅ Train pairs: {len(train_imgs)} | Test images: {len(test_imgs)}")
if len(train_imgs):
    print(f"Example train: {train_imgs[0].name} → {Path(train_gts[0]).name if train_gts[0] else None}")
if len(test_imgs):
    print(f"Example test:  {test_imgs[0].name}")


# In[5]:


import os, time, random, warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

DEVICE = torch.device("cuda" if torch.cuda.is_available() and DEVICE_FALLBACK == "cuda" else "cpu")
print("Device:", DEVICE)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# Feature extractor (normalization & resizing handled via Albumentations below)
feature_extractor = SegformerFeatureExtractor(do_resize=False, do_normalize=False)

# Albumentations transforms
train_tf = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.25),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])


# In[6]:


# Metrics
def binarize(logits, thr=0.5):
    return (torch.sigmoid(logits) > thr).float()

def confusion_components(pred, target):
    tp = (pred*target).sum().item()
    tn = ((1-pred)*(1-target)).sum().item()
    fp = (pred*(1-target)).sum().item()
    fn = ((1-pred)*target).sum().item()
    return tp, tn, fp, fn

def metrics_from_conf(tp, tn, fp, fn, eps=1e-8):
    acc  = (tp+tn) / (tp+tn+fp+fn+eps)
    prec = tp / (tp+fp+eps)
    sens = tp / (tp+fn+eps)
    spec = tn / (tn+fp+eps)
    f1   = 2*prec*sens / (prec+sens+eps)
    dice = f1  # for binary
    return dict(acc=acc, precision=prec, sensitivity=sens, specificity=spec, f1=f1, dice=dice)

# Loss
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, logits, targets, eps=1e-6):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs*targets).sum(dim=(1,2,3))
        den = (probs+targets).sum(dim=(1,2,3)) + eps
        dice = 1.0 - (num/den).mean()
        return self.bce_w*bce + self.dice_w*dice

# Model builder
def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        ENCODER_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    # Adjust decoder head to a single-channel logits if needed — here we'll output 1 channel (vessel) via a 2-class head
    # Use channel 1 as vessel logits.
    return model.to(DEVICE)

loss_fn = BCEDiceLoss()


# In[8]:


# ============================================
# SegFormer K-Fold training (self-contained fix)
# Defines RiteSegDataset + utils, then runs your CV loop
# ============================================
import warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch.optim import AdamW

import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# ---------- Defaults (use existing globals if present) ----------
try: DEVICE
except NameError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try: OUT_ROOT
except NameError:
    OUT_ROOT = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\segformer_runs")

try: FOLDS
except NameError:
    FOLDS = 5

try: SEED
except NameError:
    SEED = 42

try: BATCH_SIZE
except NameError:
    BATCH_SIZE = 2

try: NUM_WORKERS
except NameError:
    NUM_WORKERS = 0

try: LR
except NameError:
    LR = 3e-4

try: WEIGHT_DECAY
except NameError:
    WEIGHT_DECAY = 1e-4

try: MAX_EPOCHS
except NameError:
    MAX_EPOCHS = 20

try: THRESHOLD
except NameError:
    THRESHOLD = 0.5

try: ENCODER_NAME
except NameError:
    # Use the same backbone you trained with
    ENCODER_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

try: NUM_LABELS
except NameError:
    # 2 classes: background (0) and vessel (1)
    NUM_LABELS = 2

try: IMG_SIZE
except NameError:
    IMG_SIZE = (512, 512)

# ---------- Repro ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ---------- Image/mask I/O ----------
def pil_read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA",): im = im.convert("RGB")
        elif im.mode in ("L",):  im = im.convert("RGB")
        elif im.mode in ("I;16","I"):
            arr = np.array(im, dtype=np.float32); mx = arr.max() if arr.max()>0 else 1.0
            arr = (arr/mx*255.0).clip(0,255).astype(np.uint8)
            im = Image.fromarray(arr).convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        return np.array(im)

def pil_read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        arr = np.array(im)
        if arr.ndim == 3:
            arr = arr[...,0]
        return (arr > 0).astype(np.uint8)

# ---------- Albumentations ----------
train_tf = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.25),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

# ---------- Dataset ----------
class RiteSegDataset(Dataset):
    def __init__(self, img_paths, gt_paths=None, train=False):
        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.train = train
        self.tf = train_tf if train else val_tf

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        ip = self.img_paths[i]
        img = pil_read_rgb(ip)
        if self.gt_paths is not None:
            gp = self.gt_paths[i]
            mask = pil_read_mask(gp)
            aug = self.tf(image=img, mask=mask)
            x = aug["image"].float()                 # [3,H,W]
            y = aug["mask"].unsqueeze(0).float()     # [1,H,W]
            return x, y, str(ip), str(gp)
        else:
            aug = self.tf(image=img)
            x = aug["image"].float()
            return x, None, str(ip), None

# ---------- Metrics ----------
def binarize(logits, thr=THRESHOLD):
    return (torch.sigmoid(logits) > thr).float()

def confusion_components(pred, target):
    tp = (pred*target).sum().item()
    tn = ((1-pred)*(1-target)).sum().item()
    fp = (pred*(1-target)).sum().item()
    fn = ((1-pred)*target).sum().item()
    return tp, tn, fp, fn

def metrics_from_conf(tp, tn, fp, fn, eps=1e-8):
    acc  = (tp+tn) / (tp+tn+fp+fn+eps)
    prec = tp / (tp+fp+eps)
    sens = tp / (tp+fn+eps)
    spec = tn / (tn+fp+eps)
    f1   = 2*prec*sens / (prec+sens+eps)
    dice = f1  # binary
    return dict(acc=acc, precision=prec, sensitivity=sens, specificity=spec, f1=f1, dice=dice)

# ---------- Loss ----------
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w
    def forward(self, logits, targets, eps=1e-6):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs*targets).sum(dim=(1,2,3))
        den = (probs+targets).sum(dim=(1,2,3)) + eps
        dice = 1.0 - (num/den).mean()
        return self.bce_w*bce + self.dice_w*dice

loss_fn = BCEDiceLoss()

# ---------- Model ----------
def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        ENCODER_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    return model.to(DEVICE)

# ---------- Require train lists ----------
assert 'train_imgs' in globals() and 'train_gts' in globals(), \
    "Please define train_imgs and train_gts lists of Path objects before running this cell."

# ============================================
# K-Fold CV (on training split)
# ============================================
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

hist_cols = ["epoch","train_loss","train_dice","train_acc","train_prec","train_spec","train_sens","train_f1",
             "val_loss","val_dice","val_acc","val_prec","val_spec","val_sens","val_f1"]

fold_summaries = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_imgs), start=1):
    print(f"\n===== FOLD {fold}/{FOLDS} =====")
    fold_dir = OUT_ROOT / f"fold_{fold:02d}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "logs").mkdir(parents=True, exist_ok=True)
    (fold_dir / "pred_out_val").mkdir(parents=True, exist_ok=True)

    # Datasets
    tr_imgs = [train_imgs[i] for i in tr_idx]
    tr_gts  = [train_gts[i]  for i in tr_idx]
    va_imgs = [train_imgs[i] for i in val_idx]
    va_gts  = [train_gts[i]  for i in val_idx]

    ds_tr = RiteSegDataset(tr_imgs, tr_gts, train=True)
    ds_va = RiteSegDataset(va_imgs, va_gts, train=False)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_dice = -1.0
    history = []

    for epoch in range(1, MAX_EPOCHS+1):
        # ---- Train ----
        model.train()
        run_loss = 0.0
        t_metrics = dict(acc=0, precision=0, sensitivity=0, specificity=0, f1=0, dice=0)
        n_seen = 0

        for xb, yb, _, _ in dl_tr:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            out = model(pixel_values=xb).logits  # [B, C, h, w]
            if out.shape[-2:] != yb.shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=yb.shape[-2:], mode="bilinear", align_corners=False)
            vessel_logits = out[:, 1:2, :, :]     # class 1 = vessel
            loss = loss_fn(vessel_logits, yb)
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item()) * xb.size(0)

            with torch.no_grad():
                pred = binarize(vessel_logits, THRESHOLD)
                tp, tn, fp, fn = confusion_components(pred, yb)
                mets = metrics_from_conf(tp, tn, fp, fn)
                for k in t_metrics: t_metrics[k] += mets[k] * xb.size(0)
                n_seen += xb.size(0)

        train_loss = run_loss / max(1, n_seen)
        train_mets = {k: (t_metrics[k]/max(1,n_seen)) for k in t_metrics}

        # ---- Val ----
        model.eval()
        v_loss = 0.0
        v_metrics = dict(acc=0, precision=0, sensitivity=0, specificity=0, f1=0, dice=0)
        n_seen_v = 0

        with torch.no_grad():
            for xb, yb, _, _ in dl_va:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                out = model(pixel_values=xb).logits
                if out.shape[-2:] != yb.shape[-2:]:
                    out = torch.nn.functional.interpolate(out, size=yb.shape[-2:], mode="bilinear", align_corners=False)
                vessel_logits = out[:, 1:2, :, :]
                loss = loss_fn(vessel_logits, yb)
                v_loss += float(loss.item()) * xb.size(0)

                pred = binarize(vessel_logits, THRESHOLD)
                tp, tn, fp, fn = confusion_components(pred, yb)
                mets = metrics_from_conf(tp, tn, fp, fn)
                for k in v_metrics: v_metrics[k] += mets[k] * xb.size(0)
                n_seen_v += xb.size(0)

        val_loss = v_loss / max(1, n_seen_v)
        val_mets = {k: (v_metrics[k]/max(1,n_seen_v)) for k in v_metrics}

        # Save history row
        row = [
            epoch,
            train_loss, train_mets["dice"], train_mets["acc"], train_mets["precision"], train_mets["specificity"], train_mets["sensitivity"], train_mets["f1"],
            val_loss, val_mets["dice"], val_mets["acc"], val_mets["precision"], val_mets["specificity"], val_mets["sensitivity"], val_mets["f1"],
        ]
        history.append(row)
        pd.DataFrame(history, columns=hist_cols).to_csv(fold_dir / "logs" / "history.csv", index=False)

        # Save best checkpoint by val Dice
        if val_mets["dice"] > best_val_dice:
            best_val_dice = val_mets["dice"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metric": float(best_val_dice),
                "config": {"IMG_SIZE": IMG_SIZE, "THRESHOLD": THRESHOLD}
            }, fold_dir / "checkpoints" / "best_segformer.pt")
            print(f"[Fold {fold}] Epoch {epoch}: new best val Dice={best_val_dice:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            print(f"[Fold {fold}] Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"train_dice={train_mets['dice']:.4f} val_dice={val_mets['dice']:.4f}")

    fold_summaries.append({"fold": fold, "best_val_dice": best_val_dice})
    pd.DataFrame(fold_summaries).to_csv(OUT_ROOT / "cv_summary.csv", index=False)

print("\n[CV] Done. Summary:")
print(pd.DataFrame(fold_summaries))


# In[9]:


# =========================
# Plot per-fold Loss & Dice
# =========================
import matplotlib.pyplot as plt
from glob import glob

for fold in range(1, FOLDS+1):
    hist_path = OUT_ROOT / f"fold_{fold:02d}" / "logs" / "history.csv"
    if not hist_path.exists():
        continue
    hist = pd.read_csv(hist_path)
    plt.figure(figsize=(12,4), dpi=130)
    plt.subplot(1,2,1)
    plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    plt.plot(hist["epoch"], hist["val_loss"],   label="val_loss")
    plt.title(f"Fold {fold} — Loss")
    plt.xlabel("Epoch"); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(hist["epoch"], hist["train_dice"], label="train_dice")
    plt.plot(hist["epoch"], hist["val_dice"],   label="val_dice")
    plt.title(f"Fold {fold} — Dice")
    plt.xlabel("Epoch"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.show()


# In[10]:


# ============================================
# SegFormer test evaluation across folds (robust to missing GT)
# ============================================
import os, warnings, math
from time import perf_counter
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import SegformerForSemanticSegmentation

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# -----------------------------
# Defaults (use your notebook's values if already defined)
# -----------------------------
try: OUT_ROOT
except NameError: OUT_ROOT = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\segformer_runs")

try: FOLDS
except NameError: FOLDS = 5

try: THRESHOLD
except NameError: THRESHOLD = 0.5

try: NUM_WORKERS
except NameError: NUM_WORKERS = 0

try: DEVICE
except NameError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try: ENCODER_NAME
except NameError:
    # Must match your training backbone; change if you used a different SegFormer variant
    ENCODER_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

try: NUM_LABELS
except NameError:
    # 2 classes: 0=background, 1=vessel (logits[:,1] will be used for vessel)
    NUM_LABELS = 2

# =====================================================
# Utilities (robust image/mask readers, metrics, binarize)
# =====================================================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".gif"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def pil_read_rgb(path: Path) -> np.ndarray:
    """Robust RGB loader."""
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA",):
            im = im.convert("RGB")
        elif im.mode in ("L",):
            im = im.convert("RGB")
        elif im.mode in ("I;16", "I"):
            arr = np.array(im, dtype=np.float32)
            mx  = arr.max() if arr.max() > 0 else 1.0
            arr = (arr/mx*255.0).clip(0,255).astype(np.uint8)
            im  = Image.fromarray(arr).convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        return np.array(im)

def pil_read_mask(path: Path) -> np.ndarray:
    """Binary mask loader; returns 0/1 uint8."""
    with Image.open(path) as im:
        im.load()
        if im.mode not in ("L", "1"):
            im = im.convert("L")
        arr = np.array(im)
    return (arr > 0).astype(np.uint8)

def binarize(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > thr).float()

def confusion_components(pred: torch.Tensor, target: torch.Tensor):
    # pred/target shape: [B,1,H,W] with 0/1 floats
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = (((1 - pred) * target)).sum().item()
    return tp, tn, fp, fn

def metrics_from_conf(tp, tn, fp, fn, eps=1e-8):
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    f1   = 2 * prec * sens / (prec + sens + eps)
    dice = f1  # for binary case
    return dict(accuracy=acc, precision=prec, sensitivity=sens, specificity=spec, f1=f1, dice=dice)

# ============================================
# Dataset (GT optional) + simple ToTensor transform
# ============================================
import albumentations as A
from albumentations.pytorch import ToTensorV2

# If you used different normalization in training, mirror that here
IMG_SIZE = (512, 512)
val_tf = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

class RiteSegDataset(Dataset):
    """
    img_paths: list[Path]
    gt_paths:  list[Optional[Path]] or None (missing GT allowed)
    """
    def __init__(self, img_paths: List[Path], gt_paths: Optional[List[Optional[Path]]]=None):
        self.img_paths = img_paths
        self.gt_paths  = gt_paths
        self.tf        = val_tf

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, i):
        ip = self.img_paths[i]
        img = pil_read_rgb(ip)

        has_gt = self.gt_paths is not None and self.gt_paths[i] is not None
        if has_gt:
            gp   = self.gt_paths[i]
            mask = pil_read_mask(gp)
            aug  = self.tf(image=img, mask=mask)
            x    = aug["image"].float()           # [3,H,W]
            y    = aug["mask"].unsqueeze(0).float()  # [1,H,W]
            gp_s = str(gp)
        else:
            aug  = self.tf(image=img)
            x    = aug["image"].float()
            y    = None
            gp_s = None

        return x, y, str(ip), gp_s

def collate_test(batch):
    """
    Keeps y=None if any sample in the batch has no GT.
    """
    xs, ys, ips, gps = [], [], [], []
    for x, y, ip, gp in batch:
        xs.append(x); ys.append(y); ips.append(ip); gps.append(gp)
    xb = torch.stack(xs, 0)
    if any(y is None for y in ys):
        yb = None
    else:
        yb = torch.stack(ys, 0)
    return xb, yb, ips, gps

# ============================================
# Model builder (must match your training config)
# ============================================
def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        ENCODER_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,  # in case head dims differ; we load state_dict anyway
    )
    return model.to(DEVICE)

# ============================================
# Provide your test file lists here
# If you already have test_imgs/test_gts lists in memory, you can skip this section
# ============================================
def list_pairs_allow_missing(img_dir: Path, gt_dir: Optional[Path]=None):
    imgs = sorted([p for p in img_dir.rglob("*") if is_img(p)])
    gts  = None
    if gt_dir is not None and gt_dir.exists():
        gts = []
        for ip in imgs:
            cand = gt_dir / (ip.stem + ip.suffix)
            if cand.exists():
                gts.append(cand)
                continue
            found = None
            for ext in IMG_EXTS:
                c2 = gt_dir / f"{ip.stem}{ext}"
                if c2.exists():
                    found = c2; break
            gts.append(found)  # may be None
    return imgs, gts

# Example discovery (edit or replace with your own lists):
try:
    test_imgs
    test_gts
except NameError:
    TEST_IMG_DIR = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\test\img")
    TEST_GT_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\test\vessel")  # can be missing/empty
    test_imgs, test_gts = list_pairs_allow_missing(TEST_IMG_DIR, TEST_GT_DIR)

print(f"[INFO] Test images: {len(test_imgs)} (GT present for {sum(1 for g in (test_gts or []) if g is not None)} images)")

# ============================================
# DataLoader (GT optional)
# ============================================
dl_test = DataLoader(
    RiteSegDataset(test_imgs, test_gts),
    batch_size=1, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    collate_fn=collate_test
)

# ============================================
# Evaluate each fold: load best checkpoint, predict, time, save PNG, compute Dice if GT
# ============================================
OUT_ROOT.mkdir(parents=True, exist_ok=True)
all_rows = []

for fold in range(1, FOLDS + 1):
    fold_dir  = OUT_ROOT / f"fold_{fold:02d}"
    ckpt_path = fold_dir / "checkpoints" / "best_segformer.pt"
    pred_dir  = fold_dir / "pred_out_test"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        print(f"[WARN] Missing checkpoint for fold {fold}: {ckpt_path}")
        continue

    # Build & load model
    model = build_model()
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    # Accept either raw state_dict or wrapped
    state = ckpt.get("state_dict", ckpt)
    # If trained with DataParallel, keys may have "module."
    cleaned_state = {k.replace("module.", ""): v for k, v in state.items()}
    # Non-strict in case of small naming differences in head
    model.load_state_dict(cleaned_state, strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for xb, yb, ipaths, gpaths in dl_test:
            xb = xb.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = perf_counter()
            out = model(pixel_values=xb).logits
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = perf_counter()

            # Resize logits to input size if needed
            if out.shape[-2:] != xb.shape[-2:]:
                out = torch.nn.functional.interpolate(
                    out, size=xb.shape[-2:], mode="bilinear", align_corners=False
                )

            # Use channel 1 for vessel logits (assuming NUM_LABELS=2 and class 1 is vessel)
            vessel_logits = out[:, 1:2, :, :]
            pred = binarize(vessel_logits, THRESHOLD)  # [B,1,H,W] float{0,1}

            # Save prediction
            stem     = Path(ipaths[0]).stem
            out_path = pred_dir / f"{stem}_pred.png"
            pred_u8  = (pred[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(pred_u8, mode="L").save(out_path)

            # Metrics if GT provided
            dice = np.nan
            if yb is not None:
                yb = yb.to(DEVICE)
                tp, tn, fp, fn = confusion_components(pred, yb)
                mets = metrics_from_conf(tp, tn, fp, fn)
                dice = float(mets["dice"])

            rows.append({
                "fold": fold,
                "image": ipaths[0],
                "gt_path": (gpaths[0] if gpaths and gpaths[0] is not None else None),
                "pred_path": str(out_path),
                "inference_time_ms": (t1 - t0) * 1000.0,
                "dice": dice,
            })

    df_fold = pd.DataFrame(rows)
    df_fold.to_csv(fold_dir / "test_metrics_per_image.csv", index=False)
    print(f"[Fold {fold}] Saved: {fold_dir / 'test_metrics_per_image.csv'}")
    all_rows.extend(rows)

# ============================================
# Aggregate across folds
# ============================================
if all_rows:
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_ROOT / "test_metrics_all_folds.csv", index=False)
    print("\n[TEST] Mean over folds — inference_time_ms, dice:")
    print(df_all[["inference_time_ms", "dice"]].mean(numeric_only=True))
else:
    print("[TEST] No rows to aggregate (check checkpoints / test set).")


# In[11]:


# =========================
# Visualize a few test samples from Fold-1 predictions
# =========================
import matplotlib.pyplot as plt

fold_vis = 1
fold_dir = OUT_ROOT / f"fold_{fold_vis:02d}"
pred_dir = fold_dir / "pred_out_test"
csv_path = fold_dir / "test_metrics_per_image.csv"

if csv_path.exists():
    df = pd.read_csv(csv_path)
    sel = df.sample(n=min(5, len(df)), random_state=SEED)

    def _read_resized(path, hw, mask=False):
        h,w = hw
        im = Image.open(path)
        if mask:
            im = im.convert("L")
            im = im.resize((w,h), resample=Image.NEAREST)
        else:
            im = im.convert("RGB")
            im = im.resize((w,h), resample=Image.BILINEAR)
        return np.array(im)

    plt.figure(figsize=(12, 3*len(sel)), dpi=130)
    for i, row in enumerate(sel.itertuples(index=False)):
        ip = Path(row.image)
        pp = Path(row.pred_path)
        img = _read_resized(ip, IMG_SIZE, mask=False)
        pred= _read_resized(pp, IMG_SIZE, mask=True)

        plt.subplot(len(sel), 3, 3*i+1); plt.imshow(img); plt.axis("off"); plt.title(ip.name)
        # find GT if available
        gt = None
        cand = TEST_GT_DIR / (ip.stem + ip.suffix)
        if not cand.exists():
            for ext in [".png",".jpg",".jpeg",".tif",".tiff",".bmp",".ppm"]:
                c2 = TEST_GT_DIR / f"{ip.stem}{ext}"
                if c2.exists(): cand = c2; break
        if cand.exists():
            gt = _read_resized(cand, IMG_SIZE, mask=True)
            plt.subplot(len(sel), 3, 3*i+2); plt.imshow(gt, cmap="gray"); plt.axis("off"); plt.title("GT")
        else:
            plt.subplot(len(sel), 3, 3*i+2); plt.text(0.5,0.5,"No GT", ha="center", va="center"); plt.axis("off"); plt.title("GT")
        plt.subplot(len(sel), 3, 3*i+3); plt.imshow(pred, cmap="gray"); plt.axis("off"); 
        t = f"Pred (Dice={row.dice:.3f})" if not np.isnan(row.dice) else "Pred"
        plt.title(t)
    plt.tight_layout()
else:
    print(f"No CSV found at {csv_path}. Run the previous cell first.")


# In[12]:


# ============================================
# Mean ± Std Visualization across folds
# ============================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure OUT_ROOT and FOLDS are defined
OUT_ROOT = Path(OUT_ROOT)
assert OUT_ROOT.exists(), f"{OUT_ROOT} does not exist."

# Gather per-fold histories
all_hist = []
for fold in range(1, FOLDS+1):
    hist_path = OUT_ROOT / f"fold_{fold:02d}" / "logs" / "history.csv"
    if hist_path.exists():
        df = pd.read_csv(hist_path)
        df["fold"] = fold
        all_hist.append(df)
    else:
        print(f"[WARN] Missing {hist_path}")

if not all_hist:
    raise ValueError("No fold histories found!")

# Concatenate and compute mean/std grouped by epoch
df_all = pd.concat(all_hist, ignore_index=True)
metrics = ["train_loss","val_loss","train_dice","val_dice"]

mean_df = df_all.groupby("epoch")[metrics].mean()
std_df  = df_all.groupby("epoch")[metrics].std()

# ======================
# Plot with shaded std
# ======================
plt.figure(figsize=(14,6), dpi=130)

# ---- Loss subplot ----
plt.subplot(1,2,1)
epochs = mean_df.index
plt.plot(epochs, mean_df["train_loss"], label="Train Loss (mean)", color="tab:blue")
plt.fill_between(epochs,
                 mean_df["train_loss"] - std_df["train_loss"],
                 mean_df["train_loss"] + std_df["train_loss"],
                 color="tab:blue", alpha=0.2)
plt.plot(epochs, mean_df["val_loss"], label="Val Loss (mean)", color="tab:orange")
plt.fill_between(epochs,
                 mean_df["val_loss"] - std_df["val_loss"],
                 mean_df["val_loss"] + std_df["val_loss"],
                 color="tab:orange", alpha=0.2)
plt.title("Loss (Mean ± Std across folds)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# ---- Dice subplot ----
plt.subplot(1,2,2)
plt.plot(epochs, mean_df["train_dice"], label="Train Dice (mean)", color="tab:green")
plt.fill_between(epochs,
                 mean_df["train_dice"] - std_df["train_dice"],
                 mean_df["train_dice"] + std_df["train_dice"],
                 color="tab:green", alpha=0.2)
plt.plot(epochs, mean_df["val_dice"], label="Val Dice (mean)", color="tab:red")
plt.fill_between(epochs,
                 mean_df["val_dice"] - std_df["val_dice"],
                 mean_df["val_dice"] + std_df["val_dice"],
                 color="tab:red", alpha=0.2)
plt.title("Dice (Mean ± Std across folds)")
plt.xlabel("Epoch")
plt.ylabel("Dice Coefficient")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[ ]:





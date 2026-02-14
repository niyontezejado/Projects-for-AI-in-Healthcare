#!/usr/bin/env python
# coding: utf-8

# In[9]:


# ======== SETUP & CONFIG (DRIVE) ========
import os, random, warnings, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold  # ← added KFold

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch.nn.functional as F

from transformers import SegformerForSemanticSegmentation
from transformers.utils import logging as hf_logging

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
hf_logging.set_verbosity_error()  # keep HF downloads quiet

# --------- EDIT THESE PATHS TO YOUR DRIVE FOLDERS ---------
TRAIN_IMG_DIR = Path(r"C:\Users\SC\Documents\Data and Code\drive dataset\training\images")
TRAIN_MASK_DIR = Path(r"C:\Users\SC\Documents\Data and Code\drive dataset\training\1st_manual")
TEST_IMG_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\drive dataset\test\images")   # test has NO masks
OUT_ROOT      = Path(r"C:\Users\SC\Documents\Data and Code\drive dataset\outputs_segformer_drive")
# ----------------------------------------------------------

OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEED          = 42
IMG_SIZE      = (512, 512)      # (H, W)
BATCH_SIZE    = 2
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 100
THRESHOLD     = 0.5
NUM_WORKERS   = 0               # 0 on Windows is safest
PIN_MEMORY    = torch.cuda.is_available()

FOLDS         = 5               # ← added: use 5-fold cross-validation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# In[10]:


# ======== IO HELPERS (DRIVE, FINAL FIXED VERSION) ========
from pathlib import Path
import re
import numpy as np
from PIL import Image
from typing import List, Tuple

# Support all extensions found in your folders
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm", ".gif"}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def pil_read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        if im.mode != "RGB":
            im = im.convert("RGB")
        return np.array(im)

def pil_read_mask(path: Path) -> np.ndarray:
    """Return binary mask (0/1) from DRIVE labels (white vessels on black)."""
    with Image.open(path) as im:
        im.load()
        if im.mode != "L":
            im = im.convert("L")
        arr = np.array(im)
        return (arr > 127).astype(np.uint8)

# capture numeric id like 21 from '21_training', '21_manual1', etc.
_id_re = re.compile(r"(\d+)")

def _drive_key_from_image(name: str) -> str:
    """
    Example:
      '21_training.tif' → '21'
    """
    base = Path(name).stem.strip()
    base = base.replace(" ", "")
    for suf in ("_training", "_test"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
    m = _id_re.search(base)
    return m.group(1).lstrip("0") if m else base.lower()

def _drive_key_from_mask(name: str) -> str:
    """
    Example:
      '21_manual1.gif' → '21'
    """
    base = Path(name).stem.strip()
    base = base.replace(" ", "")
    for suf in ("_manual1", "_manual2"):
        if base.lower().endswith(suf):
            base = base[: -len(suf)]
    m = _id_re.search(base)
    return m.group(1).lstrip("0") if m else base.lower()

def pair_drive_training(img_dir: Path, mask_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Match DRIVE training images (21_training.tif, 22_training.tif, ...)
    with masks (21_manual1.gif, 22_manual1.gif, ...).
    """
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"TRAIN_IMG_DIR does not exist: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"TRAIN_MASK_DIR does not exist: {mask_dir}")

    imgs  = sorted([p for p in img_dir.iterdir() if is_img(p)], key=lambda p: p.name.lower())
    masks = [p for p in mask_dir.iterdir() if is_img(p)]

    print(f"[DEBUG] Found {len(imgs)} training images in {img_dir}")
    print(f"[DEBUG] Found {len(masks)} training masks  in {mask_dir}")

    # Map masks by numeric key (e.g., '21')
    mask_by_key = {_drive_key_from_mask(p.name): p for p in masks}

    pairs_img, pairs_mask, misses = [], [], []

    for ip in imgs:
        key = _drive_key_from_image(ip.name)
        mp = mask_by_key.get(key)
        if mp is None:
            misses.append((ip.name, key))
        else:
            pairs_img.append(ip)
            pairs_mask.append(mp)

    if misses:
        print(f"[WARN] {len(misses)} image(s) had no matching mask. Example:")
        for nm, k in misses[:10]:
            print(f"   image='{nm}'  -> key='{k}'  (no mask match)")
        print("[HINT] Verify that you're using 'training/1st_manual' and file names follow 'XX_training.tif' / 'XX_manual1.gif'.")

    print(f"[INFO] Successfully paired {len(pairs_img)} image-mask pairs.")
    return pairs_img, pairs_mask

def list_drive_test_images(test_img_dir: Path) -> List[Path]:
    """Return list of test image paths (e.g., 01_test.tif, 02_test.tif, ...)."""
    test_img_dir = Path(test_img_dir)
    if not test_img_dir.exists():
        raise FileNotFoundError(f"TEST_IMG_DIR does not exist: {test_img_dir}")
    imgs = sorted([p for p in test_img_dir.iterdir() if is_img(p)], key=lambda p: p.name.lower())
    print(f"[DEBUG] Found {len(imgs)} test images in {test_img_dir}")
    return imgs


# In[11]:


# ---- build splits ----
train_imgs_all, train_masks_all = pair_drive_training(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
print(f"Total training pairs after matching: {len(train_imgs_all)}")

assert len(train_imgs_all) > 0, (
    "No training pairs found. See [DEBUG]/[WARN] logs above to fix paths or filename patterns."
)
assert len(train_imgs_all) == len(train_masks_all)

# 80/20 split on *training* for train/val
from sklearn.model_selection import train_test_split
idx = np.arange(len(train_imgs_all))
train_idx, val_idx = train_test_split(idx, test_size=0.20, random_state=SEED, shuffle=True)

train_imgs = [train_imgs_all[i] for i in train_idx]
train_masks = [train_masks_all[i] for i in train_idx]
val_imgs   = [train_imgs_all[i] for i in val_idx]
val_masks  = [train_masks_all[i] for i in val_idx]

# Test set (images only)
test_imgs = list_drive_test_images(TEST_IMG_DIR)
print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test (no masks): {len(test_imgs)}")


# In[12]:


# ======== TRANSFORMS (same params) ========
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

# ======== DATASET (DRIVE) ========
class DriveDataset(Dataset):
    """
    Works with lists of image paths and (optionally) mask paths produced by your DRIVE pairing:
      - Training/Val: imgs + masks
      - Test: imgs only (masks=None)
    """
    def __init__(self, imgs, masks=None, train=False):
        self.imgs  = list(imgs)
        self.masks = list(masks) if masks is not None else None
        self.tf    = train_tf if train else val_tf

    def __len__(self): 
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        img = pil_read_rgb(ip)
        if self.masks is not None:
            mp  = self.masks[i]
            m   = pil_read_mask(mp)             # handles .gif masks -> binary {0,1}
            aug = self.tf(image=img, mask=m)
            x   = aug["image"].float()
            y   = aug["mask"].unsqueeze(0).float()  # (1,H,W) in [0,1]
        else:
            aug = self.tf(image=img)
            x   = aug["image"].float()
            y   = None
        return x, y, str(ip), (str(self.masks[i]) if self.masks is not None else None)

# ======== METRICS / LOSS (auto-pick vessel channel if num_labels=2) ========
def _vessel_logits(logits):
    # If model outputs 2 channels (background, vessel), take channel 1.
    # If it outputs 1 channel, use it directly.
    if logits.dim() == 4 and logits.size(1) >= 2:
        return logits[:, 1:2, ...]
    return logits

def binarize(logits, thr=0.5):
    logits = _vessel_logits(logits)
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
    dice = f1
    return dict(acc=acc, precision=prec, sensitivity=sens, specificity=spec, f1=f1, dice=dice)

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w, self.dice_w = bce_w, dice_w

    def forward(self, logits, targets, eps=1e-6):
        # Ensure we compute loss on the vessel channel (C=1) for consistency
        logits_v = _vessel_logits(logits)
        bce = self.bce(logits_v, targets)
        probs = torch.sigmoid(logits_v)
        num = 2.0 * (probs*targets).sum(dim=(1,2,3))
        den = (probs+targets).sum(dim=(1,2,3)) + eps
        dice = 1.0 - (num/den).mean()
        return self.bce_w*bce + self.dice_w*dice

loss_fn = BCEDiceLoss()

# ======== MODEL (same params) ========
ENCODER_NAME = "nvidia/segformer-b1-finetuned-ade-512-512"  # robust + fast
NUM_LABELS   = 2  # background + vessel (we pick channel 1 internally)

def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        ENCODER_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    return model.to(DEVICE)


# In[5]:


# ======== K-FOLD CV on TRAIN (DRIVE) ========
from torch.optim import AdamW

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
hist_cols = ["epoch","train_loss","train_dice","train_acc","train_prec","train_spec","train_sens","train_f1",
             "val_loss","val_dice","val_acc","val_prec","val_spec","val_sens","val_f1"]

# Expect these to be defined from your DRIVE split:
#   train_imgs, train_masks  (lists of Paths)
train_arr = np.array(train_imgs)
mask_arr  = np.array(train_masks)

fold_summaries = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_arr), start=1):
    print(f"\n===== FOLD {fold}/{FOLDS} =====")
    fold_dir = OUT_ROOT / f"fold_{fold:02d}"
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "logs").mkdir(parents=True, exist_ok=True)
    (fold_dir / "pred_test").mkdir(parents=True, exist_ok=True)

    tr_imgs_f = train_arr[tr_idx].tolist()
    tr_mask_f = mask_arr[tr_idx].tolist()
    va_imgs_f = train_arr[va_idx].tolist()
    va_mask_f = mask_arr[va_idx].tolist()

    # ---- swapped StareDataset -> DriveDataset (params unchanged) ----
    ds_tr = DriveDataset(tr_imgs_f, tr_mask_f, train=True)
    ds_va = DriveDataset(va_imgs_f, va_mask_f, train=False)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model     = build_model()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = -1.0
    history = []

    for epoch in range(1, MAX_EPOCHS+1):
        # ---- Train ----
        model.train()
        tot_loss, n_seen = 0.0, 0
        t_agg = dict(acc=0, precision=0, sensitivity=0, specificity=0, f1=0, dice=0)

        for xb, yb, _, _ in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(pixel_values=xb).logits
            if out.shape[-2:] != yb.shape[-2:]:
                out = F.interpolate(out, size=yb.shape[-2:], mode="bilinear", align_corners=False)
            vessel_logits = out[:, 1:2]  # keep behavior & metrics identical
            loss = loss_fn(vessel_logits, yb)
            loss.backward(); optimizer.step()

            bs = xb.size(0)
            tot_loss += float(loss.item()) * bs
            with torch.no_grad():
                pred = binarize(vessel_logits, THRESHOLD)
                tp, tn, fp, fn = confusion_components(pred, yb)
                mets = metrics_from_conf(tp, tn, fp, fn)
                for k in t_agg: t_agg[k] += mets[k] * bs
                n_seen += bs

        train_loss = tot_loss / max(1, n_seen)
        train_mets = {k: t_agg[k]/max(1,n_seen) for k in t_agg}

        # ---- Val ----
        model.eval()
        v_loss, n_seen_v = 0.0, 0
        v_agg = dict(acc=0, precision=0, sensitivity=0, specificity=0, f1=0, dice=0)

        with torch.no_grad():
            for xb, yb, _, _ in dl_va:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(pixel_values=xb).logits
                if out.shape[-2:] != yb.shape[-2:]:
                    out = F.interpolate(out, size=yb.shape[-2:], mode="bilinear", align_corners=False)
                logits = out[:, 1:2]
                loss = loss_fn(logits, yb)

                bs = xb.size(0)
                v_loss += float(loss.item()) * bs
                pred = binarize(logits, THRESHOLD)
                tp, tn, fp, fn = confusion_components(pred, yb)
                mets = metrics_from_conf(tp, tn, fp, fn)
                for k in v_agg: v_agg[k] += mets[k] * bs
                n_seen_v += bs

        val_loss = v_loss / max(1, n_seen_v)
        val_mets = {k: v_agg[k]/max(1,n_seen_v) for k in v_agg}

        row = [epoch,
               train_loss, train_mets["dice"], train_mets["acc"], train_mets["precision"], train_mets["specificity"], train_mets["sensitivity"], train_mets["f1"],
               val_loss,   val_mets["dice"],   val_mets["acc"],   val_mets["precision"],   val_mets["specificity"],   val_mets["sensitivity"],   val_mets["f1"]]
        history.append(row)
        pd.DataFrame(history, columns=hist_cols).to_csv(fold_dir / "logs" / "history.csv", index=False)

        if val_mets["dice"] > best_val:
            best_val = val_mets["dice"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metric": float(best_val),
                "config": {"IMG_SIZE": IMG_SIZE, "THRESHOLD": THRESHOLD}
            }, fold_dir / "checkpoints" / "best_segformer.pt")
            print(f"[Fold {fold}] Epoch {epoch}: new best val Dice={best_val:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            print(f"[Fold {fold}] Ep {epoch:03d} | tr_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"tr_dice={train_mets['dice']:.4f} val_dice={val_mets['dice']:.4f}")

    fold_summaries.append({"fold": fold, "best_val_dice": best_val})
    pd.DataFrame(fold_summaries).to_csv(OUT_ROOT / "cv_summary.csv", index=False)

print("\n[CV] Done.")
pd.DataFrame(fold_summaries)


# In[13]:


# ======== TEST INFERENCE (BEST OF EACH FOLD) — DRIVE, FIXED COLLATE + TIMING ========
import time

class _TestSet(Dataset):
    """
    DRIVE test set: images only (no ground truth).
    We return a zero mask tensor and empty gpath "" so DataLoader can collate.
    Downstream, we skip metrics whenever gpath == "".
    """
    def __init__(self, imgs, masks=None, tf=None):
        self.imgs  = list(imgs)
        self.masks = list(masks) if masks is not None else None  # should be None for DRIVE test
        self.tf    = tf if tf is not None else val_tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        img = pil_read_rgb(Path(ip))

        if self.masks is None:
            # Test: no GT -> return zero mask + empty gpath (so collate works)
            aug = self.tf(image=img)
            x   = aug["image"].float()             # (3,H,W)
            y   = torch.zeros(1, x.shape[1], x.shape[2], dtype=torch.float32)  # placeholder
            return x, y, str(ip), ""               # empty string signals "no GT"
        else:
            # (Not used for DRIVE test, but kept for completeness)
            mp  = self.masks[i]
            gt  = pil_read_mask(Path(mp)).astype(np.uint8)
            aug = self.tf(image=img, mask=gt)
            x   = aug["image"].float()
            y   = aug["mask"].unsqueeze(0).float()  # (1,H,W)
            return x, y, str(ip), str(mp)

def _save_pred_u8(t: torch.Tensor, path: Path):
    p = (t.detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(p, mode="L").save(path)

def _resize_np(arr: np.ndarray, hw: tuple[int,int], is_mask=False):
    H, W = hw
    im = Image.fromarray(arr.astype(np.uint8))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return np.array(im.resize((W, H), resample=resample))

all_rows = []

# Build test loader (no masks for DRIVE test)
ds_test = _TestSet(test_imgs, masks=None, tf=val_tf)
dl_test = DataLoader(ds_test, batch_size=1, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

for fold_dir in sorted(OUT_ROOT.glob("fold_*")):
    ckpt_path = fold_dir / "checkpoints" / "best_segformer.pt"
    if not ckpt_path.exists():
        print(f"[WARN] Missing checkpoint in {fold_dir}")
        continue

    # Load model
    model = build_model()
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    pred_dir = fold_dir / "pred_test"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for xb, yb, ipath, gpath in dl_test:
            xb = xb.to(DEVICE, non_blocking=True)

            # ---- timing start (sync for accurate GPU timing) ----
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(pixel_values=xb).logits
            if logits.shape[-2:] != xb.shape[-2:]:
                logits = F.interpolate(logits, size=xb.shape[-2:], mode="bilinear", align_corners=False)

            # Use vessel channel if present; otherwise assume single-channel logits
            vessel_logits = logits[:, 1:2] if logits.shape[1] > 1 else logits
            pred = binarize(vessel_logits, THRESHOLD)

            # ---- timing stop ----
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            stem = Path(ipath[0]).stem  # e.g., "01_test"
            op   = pred_dir / f"{stem}_pred.png"
            _save_pred_u8(pred[0,0].cpu(), op)

            # Skip Dice when there's no GT (gpath == "")
            dice_val = np.nan
            if gpath[0] != "":
                yb = yb.to(DEVICE)
                if yb.shape[-2:] != pred.shape[-2:]:
                    yb = F.interpolate(yb, size=pred.shape[-2:], mode="nearest")
                yb = (yb > 0.5).float()
                tp, tn, fp, fn = confusion_components(pred, yb)
                dice_val = metrics_from_conf(tp, tn, fp, fn)["dice"]

            rows.append({
                "fold": fold_dir.name,
                "image": ipath[0],
                "gt": (gpath[0] if gpath[0] != "" else None),
                "pred": str(op),
                "dice": float(dice_val) if not np.isnan(dice_val) else np.nan,
                "inference_time_ms": float(latency_ms)
            })

    df_fold = pd.DataFrame(rows)
    df_fold.to_csv(fold_dir / "test_metrics_per_image.csv", index=False)
    all_rows.extend(rows)
    print(f"[{fold_dir.name}] wrote test_metrics_per_image.csv")

# Aggregate across folds
if all_rows:
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_ROOT / "test_metrics_all_folds.csv", index=False)
    print("Saved:", OUT_ROOT / "test_metrics_all_folds.csv")


# In[6]:


# ======== TEST INFERENCE (BEST OF EACH FOLD) — DRIVE, FIXED COLLATE ========
class _TestSet(Dataset):
    """
    DRIVE test set: images only (no ground truth).
    We return a zero mask tensor and empty gpath "" so DataLoader can collate.
    Downstream, we skip metrics whenever gpath == "".
    """
    def __init__(self, imgs, masks=None, tf=None):
        self.imgs  = list(imgs)
        self.masks = list(masks) if masks is not None else None  # should be None for DRIVE test
        self.tf    = tf if tf is not None else val_tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        img = pil_read_rgb(Path(ip))

        if self.masks is None:
            # Test: no GT -> return zero mask + empty gpath (so collate works)
            aug = self.tf(image=img)
            x   = aug["image"].float()             # (3,H,W)
            y   = torch.zeros(1, x.shape[1], x.shape[2], dtype=torch.float32)  # placeholder
            return x, y, str(ip), ""               # empty string signals "no GT"
        else:
            # (Not used for DRIVE test, but kept for completeness)
            mp  = self.masks[i]
            gt  = pil_read_mask(Path(mp)).astype(np.uint8)
            aug = self.tf(image=img, mask=gt)
            x   = aug["image"].float()
            y   = aug["mask"].unsqueeze(0).float()  # (1,H,W)
            return x, y, str(ip), str(mp)

def _save_pred_u8(t: torch.Tensor, path: Path):
    p = (t.detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(p, mode="L").save(path)

def _resize_np(arr: np.ndarray, hw: tuple[int,int], is_mask=False):
    H, W = hw
    im = Image.fromarray(arr.astype(np.uint8))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return np.array(im.resize((W, H), resample=resample))

all_rows = []

# Build test loader (no masks for DRIVE test)
ds_test = _TestSet(test_imgs, masks=None, tf=val_tf)
dl_test = DataLoader(ds_test, batch_size=1, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

for fold_dir in sorted(OUT_ROOT.glob("fold_*")):
    ckpt_path = fold_dir / "checkpoints" / "best_segformer.pt"
    if not ckpt_path.exists():
        print(f"[WARN] Missing checkpoint in {fold_dir}")
        continue

    # Load model
    model = build_model()
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    pred_dir = fold_dir / "pred_test"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for xb, yb, ipath, gpath in dl_test:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(pixel_values=xb).logits
            if logits.shape[-2:] != xb.shape[-2:]:
                logits = F.interpolate(logits, size=xb.shape[-2:], mode="bilinear", align_corners=False)

            # Use existing binarize (auto-selects vessel channel if num_labels=2)
            pred = binarize(logits, THRESHOLD)

            stem = Path(ipath[0]).stem  # e.g., "01_test"
            op   = pred_dir / f"{stem}_pred.png"
            _save_pred_u8(pred[0,0].cpu(), op)

            # Skip Dice when there's no GT (gpath == "")
            dice_val = np.nan
            if gpath[0] != "":
                yb = yb.to(DEVICE)
                if yb.shape[-2:] != pred.shape[-2:]:
                    yb = F.interpolate(yb, size=pred.shape[-2:], mode="nearest")
                yb = (yb > 0.5).float()
                tp, tn, fp, fn = confusion_components(pred, yb)
                dice_val = metrics_from_conf(tp, tn, fp, fn)["dice"]

            rows.append({
                "fold": fold_dir.name,
                "image": ipath[0],
                "gt": (gpath[0] if gpath[0] != "" else None),
                "pred": str(op),
                "dice": float(dice_val) if not np.isnan(dice_val) else np.nan
            })

    df_fold = pd.DataFrame(rows)
    df_fold.to_csv(fold_dir / "test_metrics_per_image.csv", index=False)
    all_rows.extend(rows)
    print(f"[{fold_dir.name}] wrote test_metrics_per_image.csv")

# Aggregate across folds
if all_rows:
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_ROOT / "test_metrics_all_folds.csv", index=False)
    print("Saved:", OUT_ROOT / "test_metrics_all_folds.csv")


# In[7]:


# ======== PANELS PER FOLD (DRIVE) ========
def make_fold_panel(fold_dir: Path, n_examples=5, mode="bestworst"):
    csv = fold_dir / "test_metrics_per_image.csv"
    if not csv.exists():
        print(f"[SKIP] {csv} not found"); return
    df = pd.read_csv(csv)
    df["dice"] = pd.to_numeric(df.get("dice", np.nan), errors="coerce")

    # For DRIVE test, dice is often NaN (no GT). If so, fall back to random.
    have_dice = df["dice"].notna().sum()
    if mode == "bestworst" and have_dice >= max(2, n_examples):
        k1 = max(1, n_examples // 2); k2 = n_examples - k1
        sel = pd.concat([df.nsmallest(k1, "dice"), df.nlargest(k2, "dice")], ignore_index=True)
    else:
        sel = df.sample(n=min(n_examples, len(df)), random_state=42)

    H, W = IMG_SIZE
    fig, axes = plt.subplots(len(sel), 3, figsize=(12, 3.2 * len(sel)), dpi=130)
    if len(sel) == 1:
        axes = np.expand_dims(axes, 0)

    for i, r in enumerate(sel.itertuples(index=False)):
        # Image
        img = pil_read_rgb(Path(r.image))
        img = _resize_np(img, (H, W))
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        # GT (may be missing on DRIVE test)
        show_gt = isinstance(r.gt, str) and len(r.gt) > 0
        if show_gt:
            gt = pil_read_mask(Path(r.gt)).astype(np.uint8)
            gt = _resize_np(gt, (H, W), is_mask=True)
            axes[i, 1].imshow(gt, cmap="gray")
            axes[i, 1].set_title("GT")
        else:
            axes[i, 1].text(0.5, 0.5, "No GT", ha="center", va="center")
            axes[i, 1].set_title("GT")
        axes[i, 1].axis("off")

        # Prediction
        pred = np.array(Image.open(r.pred))
        pred = _resize_np(pred, (H, W), is_mask=True)
        t = "" if (pd.isna(r.dice) or not np.isfinite(r.dice)) else f" (Dice={r.dice:.3f})"
        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title(f"Prediction{t}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    outp = fold_dir / f"{fold_dir.name}_test_panel.png"
    fig.savefig(outp, bbox_inches="tight"); plt.show()
    print("Saved panel:", outp)

# Run for all folds
for fd in sorted(OUT_ROOT.glob("fold_*")):
    make_fold_panel(fd, n_examples=5, mode="bestworst")


# In[8]:


# ======== TRAINING CURVES: PER-FOLD + MEAN±SD (DRIVE) ========
def plot_mean_std(metric_col, title, ylabel, save_name):
    # collect per-fold histories aligned by epoch
    frames = []
    for fold in range(1, FOLDS+1):
        hp = OUT_ROOT / f"fold_{fold:02d}" / "logs" / "history.csv"
        if not hp.exists():
            continue
        df = pd.read_csv(hp)
        if metric_col not in df.columns:
            print(f"[WARN] {metric_col} missing in {hp}, skipping.")
            continue
        df = df[["epoch", metric_col]].copy()
        df["fold"] = fold
        frames.append(df)

    if not frames:
        print("[WARN] No histories found (or missing metric).")
        return

    all_df = pd.concat(frames, ignore_index=True)
    # wide by epoch (inner join on epoch)
    piv = all_df.pivot_table(index="epoch", columns="fold", values=metric_col)
    mean = piv.mean(axis=1)
    std  = piv.std(axis=1)

    plt.figure(figsize=(7,4), dpi=130)
    plt.plot(mean.index, mean.values, label=f"mean {metric_col}")
    plt.fill_between(mean.index, (mean-std).values, (mean+std).values, alpha=0.25, label="±1 std")
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title); plt.grid(True, alpha=0.3); plt.legend()
    outp = OUT_ROOT / save_name
    plt.savefig(outp, bbox_inches="tight"); plt.show()
    print("Saved:", outp)

# Per-fold figures (loss & dice)
for fold in range(1, FOLDS+1):
    hp = OUT_ROOT / f"fold_{fold:02d}" / "logs" / "history.csv"
    if not hp.exists():
        continue
    hist = pd.read_csv(hp)

    plt.figure(figsize=(12,4), dpi=130)

    # Loss subplot
    plt.subplot(1,2,1)
    if "train_loss" in hist.columns: plt.plot(hist["epoch"], hist["train_loss"], label="train_loss")
    if "val_loss"   in hist.columns: plt.plot(hist["epoch"], hist["val_loss"],   label="val_loss")
    plt.xlabel("Epoch"); plt.title(f"Fold {fold} — Loss"); plt.grid(True, alpha=0.3); plt.legend()

    # Dice subplot
    plt.subplot(1,2,2)
    if "train_dice" in hist.columns: plt.plot(hist["epoch"], hist["train_dice"], label="train_dice")
    if "val_dice"   in hist.columns: plt.plot(hist["epoch"], hist["val_dice"],   label="val_dice")
    plt.xlabel("Epoch"); plt.title(f"Fold {fold} — Dice"); plt.grid(True, alpha=0.3); plt.legend()

    outp = OUT_ROOT / f"fold_{fold:02d}" / "logs" / "history_plots.png"
    plt.savefig(outp, bbox_inches="tight"); plt.show()

# Mean±SD across folds
plot_mean_std("val_loss", "Validation Loss — Mean ± SD (K=5)", "Loss", "val_loss_mean_std.png")
plot_mean_std("val_dice", "Validation Dice — Mean ± SD (K=5)", "Dice", "val_dice_mean_std.png")


# In[ ]:





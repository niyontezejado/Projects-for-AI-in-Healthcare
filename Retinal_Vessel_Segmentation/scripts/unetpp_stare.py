#!/usr/bin/env python
# coding: utf-8

# 
# # U-Net++ on STARE — 70/15/15 split (Train/Val/Test)
# 
# This version uses **Train 70% / Val 15% / Test 15%**.  
# It logs **train & val loss**, **train & val Dice**, **train vs val accuracy**, and reports **test metrics** (including comparison against both AH & VK masks).
# 

# In[26]:


# If needed, install packages (run once)
# !pip install -U segmentation-models-pytorch albumentations opencv-python


# In[27]:


import os, re, glob, time, random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

# ----------------------
# Config (edit paths)
# ----------------------
DATASET_DIR = r"C:\Users\SC\Documents\Data and Code\Store dataset new"   # <— CHANGE this to your STARE root
IMG_DIR  = os.path.join(DATASET_DIR, "images")      # e.g., .ppm/.png
AH_DIR   = os.path.join(DATASET_DIR, "Labels-ah")   # AH annotation folder
VK_DIR   = os.path.join(DATASET_DIR, "labels-vk")     # VK annotation folder (optional)

OUTPUT_DIR = "./stare_results_70_15_15"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters (kept consistent with your setup)
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
IMG_SIZE = 512               # resize to square
EPOCHS = 60              # set higher (e.g., 60) for full training
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
SEED = 42

# Splits
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# In[28]:


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

def glob_multi(d, pats):
    out = []
    for p in pats:
        out.extend(glob.glob(os.path.join(d, p)))
    return sorted(out)

def stem(path): 
    return os.path.splitext(os.path.basename(path))[0]

# STARE images like 'im0001.ppm' and masks like 'im0001.ah.ppm', 'im0001.vk.ppm'
def find_pair_masks(img_path, ah_dir, vk_dir):
    s = stem(img_path)
    ah_m, vk_m = None, None
    if ah_dir and os.path.isdir(ah_dir):
        cands = glob_multi(ah_dir, [s+"*", s+".ah.*", s+".*ah*"])
        if cands: ah_m = cands[0]
    if vk_dir and os.path.isdir(vk_dir):
        cands = glob_multi(vk_dir, [s+"*", s+".vk.*", s+".*vk*"])
        if cands: vk_m = cands[0]
    return ah_m, vk_m


# In[29]:


class StareDataset(Dataset):
    def __init__(self, img_paths, ah_dir=None, vk_dir=None, transform=None, mask_source='ah', return_both=False):
        self.img_paths = img_paths
        self.ah_dir = ah_dir
        self.vk_dir = vk_dir
        self.transform = transform
        assert mask_source in ['ah','vk']
        self.mask_source = mask_source
        self.return_both = return_both

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        ip = self.img_paths[idx]
        img = np.array(Image.open(ip).convert("RGB"))          # HWC, uint8
        ah_p, vk_p = find_pair_masks(ip, self.ah_dir, self.vk_dir)

    # read masks (uint8), fall back to zeros if missing
        m_ah = np.array(Image.open(ah_p).convert("L")) if (ah_p and os.path.exists(ah_p)) else np.zeros(img.shape[:2], np.uint8)
        m_vk = np.array(Image.open(vk_p).convert("L")) if (vk_p and os.path.exists(vk_p)) else np.zeros(img.shape[:2], np.uint8)

    # binarize BEFORE resize/augs
        m_ah = (m_ah > 0).astype(np.uint8)
        m_vk = (m_vk > 0).astype(np.uint8)

        if self.transform is not None:
            aug = self.transform(image=img, mask=m_ah, mask_vk=m_vk)
            img, m_ah, m_vk = aug["image"], aug["mask"], aug["mask_vk"]

    # >>> THIS IS THE PART YOU ASKED ABOUT <<<
    # ensure image is float32 in [0,1]; masks stay {0,1} as float32
        img  = img.astype(np.float32) / 255.0                  # HWC, 0..1
        m_ah = m_ah.astype(np.float32)                         # HxW, 0/1
        m_vk = m_vk.astype(np.float32)

    # to torch tensors
        img  = torch.from_numpy(img.transpose(2,0,1))          # [3,H,W]
        m_ah = torch.from_numpy(m_ah).unsqueeze(0)             # [1,H,W]
        m_vk = torch.from_numpy(m_vk).unsqueeze(0)

        mask = m_ah if self.mask_source == 'ah' else m_vk
        sample = {"image": img, "mask": mask, "id": os.path.splitext(os.path.basename(ip))[0]}
        if self.return_both:
            sample["mask_ah"] = m_ah
            sample["mask_vk"] = m_vk
        return sample


# In[30]:


# Albumentations transforms
train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
], additional_targets={"mask_vk":"mask"})

val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
], additional_targets={"mask_vk":"mask"})


# In[31]:


# Build 70/15/15 split
img_files = glob_multi(IMG_DIR, ["*.ppm","*.png","*.jpg","*.jpeg","*.tif","*.bmp"])
assert len(img_files) > 0, "No images found—check IMG_DIR"

idxs = list(range(len(img_files)))
random.Random(SEED).shuffle(idxs)

n_total = len(idxs)
n_train = int(round(n_total * TRAIN_FRAC))
n_val   = int(round(n_total * VAL_FRAC))
# ensure all samples are used
n_test  = n_total - n_train - n_val

train_ids = idxs[:n_train]
val_ids   = idxs[n_train:n_train+n_val]
test_ids  = idxs[n_train+n_val:]

train_imgs = [img_files[i] for i in train_ids]
val_imgs   = [img_files[i] for i in val_ids]
test_imgs  = [img_files[i] for i in test_ids]

print(f"Total: {n_total} | Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")


# In[32]:


# ---- EXACT pairing for STARE ----
import glob, os, random
from torch.utils.data import DataLoader

def list_paired_images(IMG_DIR, AH_DIR, VK_DIR, seed=42):
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.ppm")))
    usable, missing = [], []
    for ip in imgs:
        base = os.path.splitext(os.path.basename(ip))[0]           # e.g. im0001
        ah = os.path.join(AH_DIR, f"{base}.ah.ppm")
        vk = os.path.join(VK_DIR, f"{base}.vk.ppm")
        if os.path.exists(ah) and os.path.exists(vk):
            usable.append(ip)
        else:
            missing.append((base, os.path.exists(ah), os.path.exists(vk)))
    print(f"Found {len(usable)} usable images; {len(missing)} with missing masks.")
    if missing:
        # show a few missing to help you fix paths/filenames
        print("Examples missing (base, has_AH, has_VK):", missing[:5])
    random.Random(seed).shuffle(usable)
    return usable

all_imgs = list_paired_images(IMG_DIR, AH_DIR, VK_DIR, seed=SEED)

# ---- 70/15/15 split ----
n = len(all_imgs)
assert n >= 3, "Not enough paired images; check your folders and filenames."
n_train = int(round(n * 0.70))
n_val   = int(round(n * 0.15))
n_test  = n - n_train - n_val

train_imgs = all_imgs[:n_train]
val_imgs   = all_imgs[n_train:n_train+n_val]
test_imgs  = all_imgs[n_train+n_val:]

print(f"Total: {n} | Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

# ---- Build datasets ----
train_ds = StareDataset(train_imgs, ah_dir=AH_DIR, vk_dir=VK_DIR,
                        transform=train_tf, mask_source='ah', return_both=False)
val_ds   = StareDataset(val_imgs,   ah_dir=AH_DIR, vk_dir=VK_DIR,
                        transform=val_tf,   mask_source='ah', return_both=True)
test_ds  = StareDataset(test_imgs,  ah_dir=AH_DIR, vk_dir=VK_DIR,
                        transform=val_tf,   mask_source='ah', return_both=True)

# ---- Use single-process loaders first (Windows-safe) ----
_workers = 0   # set to 2 or 4 AFTER this works
_pin = (torch.cuda.is_available())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=_workers, pin_memory=_pin)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=_workers, pin_memory=_pin)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=_workers, pin_memory=_pin)

# Quick sanity read (will show a real traceback if anything is wrong)
batch = next(iter(train_loader))
print("Sample batch:", batch["image"].shape, batch["mask"].shape)


# In[33]:


def build_model():
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None
    )
    return model.to(DEVICE)

model = build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5,
    min_lr=1e-6
)

bce = nn.BCEWithLogitsLoss().to(DEVICE)


# In[34]:


@torch.no_grad()
def dice_torch(prob, tgt, eps=1e-6):
    prob = (prob > 0.5).float(); tgt = (tgt > 0.5).float()
    inter = (prob*tgt).sum(dim=(2,3))
    union = prob.sum(dim=(2,3)) + tgt.sum(dim=(2,3))
    return ((2*inter + eps) / (union + eps)).mean()

@torch.no_grad()
def acc_torch(prob, tgt):
    prob = (prob > 0.5).float(); tgt = (tgt > 0.5).float()
    return (prob == tgt).float().mean()


# In[35]:


def train_one_epoch(model, loader, optimizer):
    model.train()
    n, loss_sum, dice_sum, acc_sum = 0, 0.0, 0.0, 0.0
    for batch in loader:
        imgs = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = bce(logits, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            dice_sum += dice_torch(probs, masks).item()*imgs.size(0)
            acc_sum  += acc_torch(probs, masks).item()*imgs.size(0)
        loss_sum += loss.item()*imgs.size(0); n += imgs.size(0)

    return loss_sum/n, dice_sum/n, acc_sum/n

@torch.no_grad()
def evaluate(model, loader, compare_both=True):
    model.eval()
    n, loss_sum = 0, 0.0
    dice_main = acc_main = 0.0
    dice_ah = dice_vk = acc_ah = acc_vk = 0.0

    for batch in loader:
        imgs = batch["image"].to(DEVICE, non_blocking=True)
        masks = batch["mask"].to(DEVICE, non_blocking=True)
        logits = model(imgs)
        loss = bce(logits, masks)
        probs = torch.sigmoid(logits)

        loss_sum += loss.item()*imgs.size(0); n += imgs.size(0)
        dice_main += dice_torch(probs, masks).item()*imgs.size(0)
        acc_main  += acc_torch(probs, masks).item()*imgs.size(0)

        if compare_both and ("mask_ah" in batch):
            m_ah = batch["mask_ah"].to(DEVICE)
            m_vk = batch["mask_vk"].to(DEVICE)
            dice_ah += dice_torch(probs, m_ah).item()*imgs.size(0)
            dice_vk += dice_torch(probs, m_vk).item()*imgs.size(0)
            acc_ah  += acc_torch(probs, m_ah).item()*imgs.size(0)
            acc_vk  += acc_torch(probs, m_vk).item()*imgs.size(0)

    out = {
        "loss": loss_sum/n,
        "dice": dice_main/n,
        "acc":  acc_main/n,
    }
    if compare_both and ("mask_ah" in batch):
        out.update({
            "dice_ah": dice_ah/n, "dice_vk": dice_vk/n,
            "acc_ah":  acc_ah/n,  "acc_vk":  acc_vk/n,
        })
    return out


# In[36]:


# ========= 5-FOLD CROSS-VALIDATION (with fixed 15% test holdout) =========
import os, json, numpy as np, random, glob
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

# ---- helper to list STRICT paired images (imXXXX.ppm ↔ imXXXX.ah.ppm + imXXXX.vk.ppm)
def list_paired_images(IMG_DIR, AH_DIR, VK_DIR, seed=42):
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.ppm")))
    usable, missing = [], []
    for ip in imgs:
        base = os.path.splitext(os.path.basename(ip))[0]   # e.g. im0001
        ah = os.path.join(AH_DIR, f"{base}.ah.ppm")
        vk = os.path.join(VK_DIR, f"{base}.vk.ppm")
        if os.path.exists(ah) and os.path.exists(vk):
            usable.append(ip)
        else:
            missing.append((base, os.path.exists(ah), os.path.exists(vk)))
    random.Random(seed).shuffle(usable)
    print(f"[Pairing] usable: {len(usable)} | missing: {len(missing)}")
    if missing[:3]:
        print("  Missing examples (base, has_AH, has_VK):", missing[:3])
    return usable

all_imgs = list_paired_images(IMG_DIR, AH_DIR, VK_DIR, seed=SEED)
n_total = len(all_imgs)
assert n_total >= 10, "Too few paired samples for CV—check your folders."

# ---- fixed 15% test split; 5-fold on remaining 85%
n_test = int(round(n_total * 0.15))
test_imgs = all_imgs[:n_test]
cv_pool   = all_imgs[n_test:]
print(f"TOTAL={n_total} | TEST={len(test_imgs)} | CV-POOL={len(cv_pool)}")

# ---- test loader once
test_ds = StareDataset(test_imgs, ah_dir=AH_DIR, vk_dir=VK_DIR,
                       transform=val_tf, mask_source='ah', return_both=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=(torch.cuda.is_available()))

# ---- CV setup
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_summaries = []
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fold, (tr_idx, va_idx) in enumerate(kf.split(cv_pool), start=1):
    print(f"\n========== Fold {fold}/5 ==========")
    train_imgs = [cv_pool[i] for i in tr_idx]
    val_imgs   = [cv_pool[i] for i in va_idx]
    print(f"Fold {fold}: train {len(train_imgs)} | val {len(val_imgs)}")

    # Datasets/loaders (Windows-safe: start with num_workers=0, raise later)
    train_ds = StareDataset(train_imgs, ah_dir=AH_DIR, vk_dir=VK_DIR,
                            transform=train_tf, mask_source='ah', return_both=False)
    val_ds   = StareDataset(val_imgs,   ah_dir=AH_DIR, vk_dir=VK_DIR,
                            transform=val_tf,   mask_source='ah', return_both=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(torch.cuda.is_available()))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=(torch.cuda.is_available()))

    # Model/optim/scheduler per fold
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6
    )
    bpath = os.path.join(OUTPUT_DIR, f"unetpp_stare_best_fold{fold}.pth")

    # history per fold
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_acc":  [], "val_acc":  [],
        "val_dice_ah": [], "val_dice_vk": [],
        "val_acc_ah":  [], "val_acc_vk":  [],
        "lr": []
    }

    best_val = float("inf")

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_dice, tr_acc = train_one_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader, compare_both=True)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["train_acc"].append(tr_acc)

        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_dice_ah"].append(val_metrics.get("dice_ah", np.nan))
        history["val_dice_vk"].append(val_metrics.get("dice_vk", np.nan))
        history["val_acc_ah"].append(val_metrics.get("acc_ah", np.nan))
        history["val_acc_vk"].append(val_metrics.get("acc_vk", np.nan))
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, bpath)

        t = time.time() - t0
        print(f"Fold {fold} | Epoch {epoch:02d}/{EPOCHS} | "
              f"Train L {tr_loss:.4f} D {tr_dice:.3f} A {tr_acc:.3f} | "
              f"Val L {val_metrics['loss']:.4f} D {val_metrics['dice']:.3f} A {val_metrics['acc']:.3f} | "
              f"AH/VK D {val_metrics.get('dice_ah',float('nan')):.3f}/{val_metrics.get('dice_vk',float('nan')):.3f} | "
              f"time {t:.1f}s")

    # save per-fold history
    hist_path = os.path.join(OUTPUT_DIR, f"cv_fold{fold}_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"[Fold {fold}] history saved -> {hist_path}")
    print(f"[Fold {fold}] best model -> {bpath}")

    # ---- evaluate best ckpt on TEST set
    ckpt = torch.load(bpath, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, compare_both=True)
    fold_summaries.append({"fold": fold, **test_metrics})
    print(f"[Fold {fold}] TEST:", {k: round(v,4) for k,v in test_metrics.items()})

# ---- aggregate test metrics across folds
keys = fold_summaries[0].keys() - {"fold"}
agg = {}
for k in keys:
    vals = np.array([fs[k] for fs in fold_summaries], dtype=float)
    agg[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

print("\n===== 5-FOLD TEST SUMMARY (mean ± std) =====")
for k, v in agg.items():
    print(f"{k:>10}: {v['mean']:.4f} ± {v['std']:.4f}")

with open(os.path.join(OUTPUT_DIR, "cv5_test_summary.json"), "w") as f:
    json.dump({"per_fold": fold_summaries, "aggregate": agg}, f, indent=2)
print("Saved:", os.path.join(OUTPUT_DIR, "cv5_test_summary.json"))


# In[37]:


# ========= CV-aware plots (works for both CV and single run) =========
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _plot_simple(history):
    """Fallback: single-run plots using in-memory `history` dict."""
    def _plot(series_dict, title, ylabel):
        plt.figure(figsize=(6,4))
        for label, series in series_dict.items():
            if series is None: 
                continue
            plt.plot(series, label=label)
        plt.title(title); plt.xlabel("epoch"); plt.ylabel(ylabel)
        plt.legend(); plt.grid(True); plt.show()

    _plot({"Train": history["train_loss"], "Val": history["val_loss"]}, "Loss", "BCE+Logits")
    _plot({"Train": history["train_dice"], "Val": history["val_dice"]}, "Dice", "Dice")
    _plot({"Train": history["train_acc"],  "Val": history["val_acc"]},  "Accuracy", "Accuracy")

def _load_cv_histories(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "cv_fold*_history.csv")))
    hists = [pd.read_csv(f) for f in files]
    return files, hists

def _plot_per_fold(histories, train_key, val_key, title, ylabel):
    plt.figure(figsize=(7,4))
    for i, df in enumerate(histories, 1):
        plt.plot(df[train_key].values, alpha=0.5, label=f"Train F{i}")
        plt.plot(df[val_key].values,  alpha=0.9, linestyle="--", label=f"Val F{i}")
    plt.title(f"{title} (per fold)")
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.legend(ncol=2)
    plt.grid(True); plt.show()

def _plot_mean_std(histories, train_key, val_key, title, ylabel):
    # Truncate to the shortest run length to aggregate safely
    min_len = min(len(df) for df in histories)
    train_stack = np.stack([df[train_key].values[:min_len] for df in histories], axis=0)
    val_stack   = np.stack([df[val_key].values[:min_len]   for df in histories], axis=0)

    train_mean = train_stack.mean(0); train_std = train_stack.std(0, ddof=1)
    val_mean   = val_stack.mean(0);   val_std   = val_stack.std(0, ddof=1)

    x = np.arange(1, min_len+1)
    plt.figure(figsize=(7,4))
    plt.plot(x, train_mean, label="Train (mean)")
    plt.fill_between(x, train_mean-train_std, train_mean+train_std, alpha=0.2, label="Train ±1σ")
    plt.plot(x, val_mean, label="Val (mean)")
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, alpha=0.2, label="Val ±1σ")
    plt.title(f"{title} (mean ± std across folds)")
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.legend(); plt.grid(True); plt.show()

# ---- Try CV plots first; if no CV histories exist, fall back to single-run ----
cv_files, cv_histories = _load_cv_histories(OUTPUT_DIR)
if len(cv_histories) >= 2:
    # Per-fold curves
    _plot_per_fold(cv_histories, "train_loss", "val_loss", "Loss", "BCE+Logits")
    _plot_per_fold(cv_histories, "train_dice", "val_dice", "Dice", "Dice")
    _plot_per_fold(cv_histories, "train_acc",  "val_acc",  "Accuracy", "Accuracy")

    # Mean ± std curves
    _plot_mean_std(cv_histories, "train_loss", "val_loss", "Loss", "BCE+Logits")
    _plot_mean_std(cv_histories, "train_dice", "val_dice", "Dice", "Dice")
    _plot_mean_std(cv_histories, "train_acc",  "val_acc",  "Accuracy", "Accuracy")

    # (Optional) AH vs VK agreement on validation — mean ± std
    if "val_dice_ah" in cv_histories[0].columns and "val_dice_vk" in cv_histories[0].columns:
        _plot_per_fold(cv_histories, "val_dice_ah", "val_dice_vk", "Val Dice (AH vs VK)", "Dice")
        _plot_mean_std(cv_histories, "val_dice_ah", "val_dice_vk", "Val Dice (AH vs VK)", "Dice")
else:
    # Single run fallback using the existing `history` dict
    _plot_simple(history)

# ---- (Optional) Visualize test summary if you saved cv5_test_summary.json ----
summary_path = os.path.join(OUTPUT_DIR, "cv5_test_summary.json")
if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)
    print("Per-fold TEST metrics:")
    for row in summary["per_fold"]:
        print(row)
    print("\nAggregate TEST metrics (mean ± std):")
    for k, v in summary["aggregate"].items():
        print(f"{k:>12}: {v['mean']:.4f} ± {v['std']:.4f}")


# In[38]:


# ==== Test evaluation (works for 5-fold CV or single run) ====
import glob, json, numpy as np

fold_ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "unetpp_stare_best_fold*.pth")))
results = []

if len(fold_ckpts) >= 2:
    print(f"Found {len(fold_ckpts)} fold checkpoints — evaluating on TEST set...")
    for i, ck in enumerate(fold_ckpts, 1):
        # fresh model per fold
        model = build_model()
        ckpt = torch.load(ck, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])

        metrics = evaluate(model, test_loader, compare_both=True)
        results.append({"fold": i, **{k: float(v) for k, v in metrics.items()}})
        print(f"[Fold {i}] TEST:", {k: f"{v:.4f}" for k, v in metrics.items()})

    # Aggregate mean ± std across folds
    keys = [k for k in results[0].keys() if k != "fold"]
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in results], dtype=float)
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}

    # Save CV summary
    with open(os.path.join(OUTPUT_DIR, "cv5_test_summary.json"), "w") as f:
        json.dump({"per_fold": results, "aggregate": agg}, f, indent=2)

    print("\n===== 5-FOLD TEST SUMMARY (mean ± std) =====")
    for k, v in agg.items():
        print(f"{k:>12}: {v['mean']:.4f} ± {v['std']:.4f}")

else:
    # ---- Single-run fallback (no CV checkpoints found) ----
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    metrics = evaluate(model, test_loader, compare_both=True)
    print("TEST METRICS:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save test metrics
    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


# In[39]:


# ==== CV-aware visualization (best fold OR ensemble) with ORIGINAL images ====
import os, glob, json, numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# --- helper: make arrays safe for imshow (float 0..1) ---
@torch.no_grad()
def _prepare_for_show(x_np):
    x = x_np.astype(np.float32)
    if x.max() > 1.0:  # e.g., uint8 0..255
        x /= 255.0
    return np.clip(x, 0.0, 1.0)

# ---------- helper: load best single fold checkpoint ----------
def load_best_fold_model(select_metric="dice"):
    """
    Picks the best fold by `select_metric` ('dice' or 'acc' -> max; 'loss' -> min).
    Uses cv5_test_summary.json if present; otherwise evaluates all fold ckpts on TEST.
    Returns: (model, best_fold_idx, metrics_dict)
    """
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "unetpp_stare_best_fold*.pth")))
    assert len(ckpts) >= 1, "No fold checkpoints found in OUTPUT_DIR."

    summary_path = os.path.join(OUTPUT_DIR, "cv5_test_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        per_fold = summary["per_fold"]
    else:
        # evaluate folds now to decide the best
        per_fold = []
        print("[Viz] cv5_test_summary.json not found, evaluating folds on TEST...")
        for i, ck in enumerate(ckpts, 1):
            m = build_model()
            m.load_state_dict(torch.load(ck, map_location=DEVICE)["state_dict"])
            m.to(DEVICE).eval()
            met = evaluate(m, test_loader, compare_both=True)  # uses your existing evaluate()
            per_fold.append({"fold": i, **{k: float(v) for k, v in met.items()}})
            print(f"  Fold {i}: dice={met['dice']:.4f} loss={met['loss']:.4f} acc={met['acc']:.4f}")

    # choose max for dice/acc, min for loss
    key = select_metric.lower()
    if key == "loss":
        best = min(per_fold, key=lambda r: r.get(key, float("inf")))
    else:
        best = max(per_fold, key=lambda r: r.get(key, -float("inf")))

    best_fold = int(best["fold"])
    best_ckpt = os.path.join(OUTPUT_DIR, f"unetpp_stare_best_fold{best_fold}.pth")

    model = build_model()
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE)["state_dict"])
    model.to(DEVICE).eval()
    print(f"[Viz] Using BEST FOLD {best_fold} by '{select_metric}':",
          {k: round(v,4) for k,v in best.items() if k != "fold"})
    return model, best_fold, best

# ---------- helper: load ALL fold models (for ensembling) ----------
def load_all_fold_models():
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "unetpp_stare_best_fold*.pth")))
    assert len(ckpts) >= 1, "No fold checkpoints found in OUTPUT_DIR."
    models = []
    for ck in ckpts:
        m = build_model()
        m.load_state_dict(torch.load(ck, map_location=DEVICE)["state_dict"])
        m.to(DEVICE).eval()
        models.append(m)
    print(f"[Viz] Loaded {len(models)} fold models for ensembling.")
    return models

# ---------- single-model visualization (best fold) ----------
@torch.no_grad()
def visualize_best_fold(loader, n=5, thr=0.5, select_metric="dice"):
    model, best_fold, _ = load_best_fold_model(select_metric=select_metric)
    shown = 0
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        ids  = batch["id"]  # stems like 'im0291'
        logits = model(imgs)
        probs  = torch.sigmoid(logits)
        preds  = (probs > thr).float().cpu().numpy()
        m_ah   = batch.get("mask_ah", batch["mask"]).cpu().numpy()
        m_vk   = batch.get("mask_vk", batch["mask"]).cpu().numpy()

        # model inputs (for fallback)
        imgs_np = imgs.detach().cpu().numpy().transpose(0,2,3,1)

        for i in range(imgs.size(0)):
            if shown >= n:
                return

            # load the TRUE original from disk if mapping exists; otherwise fallback
            orig_path = None
            if "ID2PATH_TEST" in globals():
                orig_path = ID2PATH_TEST.get(ids[i])
            orig_show = _prepare_for_show(
                np.array(Image.open(orig_path).convert("RGB")) if orig_path else imgs_np[i]
            )
            orig_title = f"Original: {os.path.basename(orig_path)} (best fold {best_fold})" if orig_path \
                         else f"Input (best fold {best_fold})"

            fig, axs = plt.subplots(1, 4, figsize=(12,3))
            axs[0].imshow(orig_show);               axs[0].set_title(orig_title); axs[0].axis("off")
            axs[1].imshow(preds[i,0], cmap="gray"); axs[1].set_title("Pred");     axs[1].axis("off")
            axs[2].imshow(m_ah[i,0], cmap="gray");  axs[2].set_title("GT AH");    axs[2].axis("off")
            axs[3].imshow(m_vk[i,0], cmap="gray");  axs[3].set_title("GT VK");    axs[3].axis("off")
            plt.show()
            shown += 1

# ---------- ensemble visualization (average logits from all folds) ----------
@torch.no_grad()
def visualize_ensemble(loader, n=5, thr=0.5):
    models = load_all_fold_models()
    shown = 0
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        ids  = batch["id"]
        # average logits across folds
        logits_sum = None
        for m in models:
            out = m(imgs)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        logits = logits_sum / len(models)
        probs  = torch.sigmoid(logits)
        preds  = (probs > thr).float().cpu().numpy()
        m_ah   = batch.get("mask_ah", batch["mask"]).cpu().numpy()
        m_vk   = batch.get("mask_vk", batch["mask"]).cpu().numpy()
        imgs_np = imgs.detach().cpu().numpy().transpose(0,2,3,1)

        for i in range(imgs.size(0)):
            if shown >= n:
                return
            orig_path = None
            if "ID2PATH_TEST" in globals():
                orig_path = ID2PATH_TEST.get(ids[i])
            orig_show = _prepare_for_show(
                np.array(Image.open(orig_path).convert("RGB")) if orig_path else imgs_np[i]
            )
            fig, axs = plt.subplots(1, 4, figsize=(12,3))
            axs[0].imshow(orig_show);               axs[0].set_title("Original | ensemble"); axs[0].axis("off")
            axs[1].imshow(preds[i,0], cmap="gray"); axs[1].set_title("Pred (ensemble)");    axs[1].axis("off")
            axs[2].imshow(m_ah[i,0], cmap="gray");  axs[2].set_title("GT AH");              axs[2].axis("off")
            axs[3].imshow(m_vk[i,0], cmap="gray");  axs[3].set_title("GT VK");              axs[3].axis("off")
            plt.show()
            shown += 1

# ---- Call ONE of the following:
visualize_best_fold(test_loader, n=5, thr=0.5, select_metric="dice")
# visualize_ensemble(test_loader, n=5, thr=0.5)


# In[ ]:





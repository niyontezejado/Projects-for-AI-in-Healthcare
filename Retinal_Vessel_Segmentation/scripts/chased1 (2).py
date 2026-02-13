#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==== CHASE-DB1: use local folders (single-mask) ====
import os, random
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

SEED = 42
random.Random(SEED)

# --- IMPORTANT: use raw strings r"..."
IMAGES_DIR = Path(r"C:\Users\SC\Documents\Data and Code\d1\Images")
MASKS_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\d1\Masks")

assert IMAGES_DIR.exists(), f"Images dir not found: {IMAGES_DIR}"
assert MASKS_DIR.exists(),  f"Masks  dir not found: {MASKS_DIR}"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}

def is_img(p: Path):
    return p.is_file() and p.suffix.lower() in IMG_EXTS

# tokens commonly used in CHASE-DB1 masks
MASK_SUFFIXES = ["_1stho", "_2ndho", "_manual1", "_manual2", "-mask", "_mask", "-label", "_label"]

# gather images & masks from their separate roots
all_imgs  = sorted([p for p in IMAGES_DIR.rglob("*") if is_img(p)])
all_masks = sorted([p for p in MASKS_DIR.rglob("*")  if is_img(p)])
print(f"Found {len(all_imgs)} images | {len(all_masks)} masks")

# index masks by several keys (exact stem and stem without common suffixes), case-insensitive
mask_index = defaultdict(list)
for m in all_masks:
    stem_orig = m.stem              # original case
    stem_low  = stem_orig.lower()   # lower for suffix removal

    # exact stems
    mask_index[stem_orig].append(m)
    mask_index[stem_low].append(m)

    # stems with common suffixes removed
    for suf in MASK_SUFFIXES:
        if stem_low.endswith(suf):
            base = stem_orig[:len(stem_orig) - len(suf)]
            base_low = base.lower()
            mask_index[base].append(m)
            mask_index[base_low].append(m)

# pair each image to ONE mask (single-mask setup)
pairs, missing = [], []
for ip in all_imgs:
    stem_orig = ip.stem
    stem_low  = stem_orig.lower()
    cands = mask_index.get(stem_orig, []) + mask_index.get(stem_low, [])
    if cands:
        # pick the first candidate (you can add a preference rule if you want 1stHO over others)
        pairs.append({"img": ip, "mask": sorted(set(cands))[0]})
    else:
        missing.append(ip)

print(f"Usable pairs: {len(pairs)} | Images without mask: {len(missing)}")
if missing:
    # show a few to debug naming if needed
    print("Examples missing:", [m.name for m in missing[:5]])

# 70 / 15 / 15 split (deterministic)
random.Random(SEED).shuffle(pairs)
n = len(pairs)
n_train = int(round(n * 0.70))
n_val   = int(round(n * 0.15))
n_test  = n - n_train - n_val

train_recs = pairs[:n_train]
val_recs   = pairs[n_train:n_train+n_val]
test_recs  = pairs[n_train+n_val:]

print(f"SPLIT -> train {len(train_recs)} | val {len(val_recs)} | test {len(test_recs)}")

# map id -> original image path for visualization
ID2PATH_TEST = {d["img"].stem: str(d["img"]) for d in test_recs}


# In[2]:


# ==== Dataset (single-mask) ====
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, torch
from torch.utils.data import Dataset, DataLoader

IMG_SIZE      = 512
BATCH_SIZE    = 2  # same as before

class ChaseDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = np.array(Image.open(rec["img"]).convert("RGB"))
        msk = np.array(Image.open(rec["mask"]).convert("L"))
        msk = (msk > 0).astype(np.uint8)

        if self.transform is not None:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]

        img = img.astype(np.float32)/255.0
        msk = msk.astype(np.float32)

        img_t = torch.from_numpy(img.transpose(2,0,1))
        msk_t = torch.from_numpy(msk).unsqueeze(0)
        return {"image": img_t, "mask": msk_t, "id": rec["img"].stem}

# transforms
train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
])
val_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
])

# data loaders (num_workers=0 is safest on Windows to avoid worker crashes)
pin = torch.cuda.is_available()
train_loader = DataLoader(ChaseDataset(train_recs, transform=train_tf), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
val_loader   = DataLoader(ChaseDataset(val_recs,   transform=val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
test_loader  = DataLoader(ChaseDataset(test_recs,  transform=val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

b = next(iter(train_loader))
print("Batch shapes:", tuple(b["image"].shape), tuple(b["mask"].shape))


# In[3]:


# ==== Quick visual sanity check: show 3 samples (image, mask, overlay) ====
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def _load_pair(rec):
    """Load RGB image and binarized mask for a single record dict {'img': Path, 'mask': Path}."""
    img = np.array(Image.open(rec["img"]).convert("RGB"))
    msk = np.array(Image.open(rec["mask"]).convert("L"))
    msk = (msk > 0).astype(np.uint8)  # binarize to {0,1}
    return img, msk

def _overlay_mask(img, msk, color=(255, 0, 0), alpha=0.4):
    """Overlay binary mask on image with a given color and alpha."""
    ov = img.copy().astype(np.float32)
    col = np.array(color, dtype=np.float32)
    # apply to masked pixels
    mask_idx = msk.astype(bool)
    ov[mask_idx] = (1 - alpha) * ov[mask_idx] + alpha * col
    return ov.clip(0, 255).astype(np.uint8)

def show_samples(records, n=3, seed=42, title_prefix="train"):
    """Show n samples (image, mask, overlay)."""
    n = min(n, len(records))
    rng = random.Random(seed)
    picks = rng.sample(records, k=n)

    for rec in picks:
        img, msk = _load_pair(rec)
        ov = _overlay_mask(img, msk, color=(255, 0, 0), alpha=0.45)

        fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
        axs[0].imshow(img);        axs[0].set_title(f"{title_prefix} | {rec['img'].stem}"); axs[0].axis("off")
        axs[1].imshow(msk, cmap="gray"); axs[1].set_title("Mask (binary)"); axs[1].axis("off")
        axs[2].imshow(ov);         axs[2].set_title("Overlay"); axs[2].axis("off")
        plt.tight_layout()
        plt.show()

# ---- Call it (run one or more of these) ----
show_samples(train_recs, n=3, seed=42, title_prefix="train")
# show_samples(val_recs,   n=3, seed=43, title_prefix="val")
# show_samples(test_recs,  n=3, seed=44, title_prefix="test")


# In[4]:


# ======== 5-Fold Cross-Validation (same params) ========
import os, time, json, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --- Reuse your globals if already defined; otherwise set defaults
SEED          = globals().get("SEED", 42)
FOLDS         = 5
DEVICE        = globals().get("DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
EPOCHS        = globals().get("EPOCHS", 60)
LEARNING_RATE = globals().get("LEARNING_RATE", 1e-4)
ACC_THR       = globals().get("ACC_THR", 0.5)
OUTPUT_DIR    = globals().get("OUTPUT_DIR", "./chase_runs")
IMG_SIZE      = globals().get("IMG_SIZE", 512)
BATCH_SIZE    = globals().get("BATCH_SIZE", 2)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Use all pairs for CV (if you had split earlier, we recombine)
if 'pairs' in globals():
    pairs_all = list(pairs)  # full paired list from your discovery cell
elif all(k in globals() for k in ['train_recs','val_recs','test_recs']):
    pairs_all = list(train_recs) + list(val_recs) + list(test_recs)
else:
    raise RuntimeError("Could not find 'pairs' or (train_recs, val_recs, test_recs). Run the pairing cell first.")

# --- Safety: make sure transforms & dataset class exist (from your earlier cells)
if 'ChaseDataset' not in globals():
    raise RuntimeError("ChaseDataset class not found. Define it before running CV.")
if 'train_tf' not in globals() or 'val_tf' not in globals():
    raise RuntimeError("Albumentations transforms 'train_tf'/'val_tf' not found. Define them before running CV.")

pin = torch.cuda.is_available()

def make_loaders_from_indices(idx_train, idx_val):
    trs = [pairs_all[i] for i in idx_train]
    vrs = [pairs_all[i] for i in idx_val]
    train_loader = DataLoader(ChaseDataset(trs, transform=train_tf), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(ChaseDataset(vrs, transform=val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
    return train_loader, val_loader

# ---- Metrics & loss (reuse from your earlier definitions if present)
if 'criterion' not in globals() or 'dice_coef' not in globals() or 'accuracy' not in globals():
    import torch.nn as nn
    import segmentation_models_pytorch as smp
    bce_loss = nn.BCEWithLogitsLoss().to(DEVICE)
    def dice_coef(pred, target, eps=1e-7):
        prob = torch.sigmoid(pred)
        num = 2.0*(prob*target).sum(dim=(2,3))
        den = (prob+target).sum(dim=(2,3)) + eps
        return (num/den).mean()
    def accuracy(pred, target, thr=ACC_THR):
        prob = torch.sigmoid(pred)
        predb = (prob > thr).float()
        return (predb == target).float().mean()
    def criterion(pred, target, bce_w=0.5):
        bce = bce_loss(pred, target)
        dsc = 1.0 - dice_coef(pred, target)
        return bce_w*bce + (1.0 - bce_w)*dsc
    def build_model():
        return smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        ).to(DEVICE)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tloss, tdice, tacc = [], [], []
    for batch in loader:
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)
        out = model(img)
        loss = criterion(out, msk)
        tloss.append(loss.item())
        tdice.append(dice_coef(out, msk).item())
        tacc.append(accuracy(out, msk).item())
    return {"loss": float(np.mean(tloss)), "dice": float(np.mean(tdice)), "acc": float(np.mean(tacc))}

def train_one_epoch(model, loader, optimizer):
    model.train()
    tloss, tdice, tacc = [], [], []
    for batch in loader:
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, msk)
        loss.backward()
        optimizer.step()
        tloss.append(loss.item())
        tdice.append(dice_coef(out, msk).item())
        tacc.append(accuracy(out, msk).item())
    return float(np.mean(tloss)), float(np.mean(tdice)), float(np.mean(tacc))

# ---- Build fold indices (manual, no sklearn dependency)
N = len(pairs_all)
indices = list(range(N))
random.Random(SEED).shuffle(indices)
fold_bins = np.array_split(indices, FOLDS)

cv_summary = {"per_fold": [], "mean": {}}

for fold_idx in range(FOLDS):
    print(f"\n========== Fold {fold_idx+1}/{FOLDS} ==========")
    # Reproducibility per fold (optional)
    torch.manual_seed(SEED + fold_idx); torch.cuda.manual_seed_all(SEED + fold_idx); np.random.seed(SEED + fold_idx); random.seed(SEED + fold_idx)

    val_idx = list(fold_bins[fold_idx])
    train_idx = [i for f in range(FOLDS) if f != fold_idx for i in fold_bins[f]]

    train_loader, val_loader = make_loaders_from_indices(train_idx, val_idx)

    model     = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "train_acc": [], "val_acc": [], "lr": []}

    best_val = float("inf")
    best_path = os.path.join(OUTPUT_DIR, f"unetpp_chase_best_fold{fold_idx+1}.pth")

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_dice, tr_acc = train_one_epoch(model, train_loader, optimizer)
        val_metrics = evaluate(model, val_loader)
        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_acc"].append(val_metrics["acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, best_path)

        dt = time.time() - t0
        print(f"Fold {fold_idx+1} | Epoch {epoch:02d}/{EPOCHS} | "
              f"Train L {tr_loss:.4f} D {tr_dice:.3f} A {tr_acc:.3f} | "
              f"Val L {val_metrics['loss']:.4f} D {val_metrics['dice']:.3f} A {val_metrics['acc']:.3f} | "
              f"time {dt:.1f}s")

    # Save per-fold history
    hist_csv = os.path.join(OUTPUT_DIR, f"history_fold{fold_idx+1}.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)
    print(f"✓ Saved {hist_csv}")
    print(f"✓ Best checkpoint -> {best_path}")

    # Evaluate the BEST checkpoint on this fold's validation set
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    best_val_metrics = evaluate(model, val_loader)

    cv_summary["per_fold"].append({
        "fold": fold_idx+1,
        "loss": float(best_val_metrics["loss"]),
        "dice": float(best_val_metrics["dice"]),
        "acc":  float(best_val_metrics["acc"]),
        "best_ckpt": os.path.basename(best_path)
    })

    # free GPU a bit between folds
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---- Aggregate and save summary
mean_loss = float(np.mean([r["loss"] for r in cv_summary["per_fold"]]))
mean_dice = float(np.mean([r["dice"] for r in cv_summary["per_fold"]]))
mean_acc  = float(np.mean([r["acc"]  for r in cv_summary["per_fold"]]))
cv_summary["mean"] = {"loss": mean_loss, "dice": mean_dice, "acc": mean_acc}

with open(os.path.join(OUTPUT_DIR, "cv5_val_summary.json"), "w") as f:
    json.dump(cv_summary, f, indent=2)
print("\n===== CV 5-Fold Summary =====")
print(json.dumps(cv_summary, indent=2))
print("✓ Saved cv5_val_summary.json")


# In[5]:


# ========= Test Evaluation (CV-aware) =========
import os, glob, json
import numpy as np
import torch

def _list_fold_ckpts(output_dir):
    return sorted(glob.glob(os.path.join(output_dir, "unetpp_chase_best_fold*.pth")))

def _load_model_from_ckpt(ckpt_path):
    m = build_model()
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["state_dict"])
    m.to(DEVICE).eval()
    return m

@torch.no_grad()
def evaluate_ensemble(models, loader):
    """
    Average logits from all models, then compute the same loss/metrics.
    """
    tloss, tdice, tacc = [], [], []
    for batch in loader:
        imgs  = batch["image"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)

        logits_sum = None
        for m in models:
            out = m(imgs)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        logits = logits_sum / len(models)

        loss = criterion(logits, masks)
        tloss.append(loss.item())
        tdice.append(dice_coef(logits, masks).item())
        tacc.append(accuracy(logits, masks).item())

    return {
        "loss": float(np.mean(tloss)),
        "dice": float(np.mean(tdice)),
        "acc":  float(np.mean(tacc)),
    }

fold_ckpts = _list_fold_ckpts(OUTPUT_DIR)

if len(fold_ckpts) > 0:
    print(f"Found {len(fold_ckpts)} fold checkpoints. Evaluating each on TEST...")

    # Per-fold metrics
    per_fold = []
    for i, ck in enumerate(fold_ckpts, 1):
        model_i = _load_model_from_ckpt(ck)
        m_i = evaluate(model_i, test_loader)
        rec = {
            "fold": i,
            "ckpt": os.path.basename(ck),
            "loss": float(m_i["loss"]),
            "dice": float(m_i["dice"]),
            "acc":  float(m_i["acc"]),
        }
        per_fold.append(rec)
        print(f"  Fold {i}: loss={rec['loss']:.4f} dice={rec['dice']:.4f} acc={rec['acc']:.4f}")

    # Choose "best" fold (by Dice; change to 'loss' if desired)
    SELECT_METRIC = "dice"   # or "loss"
    if SELECT_METRIC == "loss":
        best_fold_rec = min(per_fold, key=lambda r: r["loss"])
    else:
        best_fold_rec = max(per_fold, key=lambda r: r["dice"])

    # Evaluate ensemble (average logits across all folds)
    models = [_load_model_from_ckpt(ck) for ck in fold_ckpts]
    ens_metrics = evaluate_ensemble(models, test_loader)

    # Aggregate & save
    cv_summary = {
        "per_fold": per_fold,
        "best_fold": best_fold_rec,
        "ensemble": {
            "loss": float(ens_metrics["loss"]),
            "dice": float(ens_metrics["dice"]),
            "acc":  float(ens_metrics["acc"]),
        }
    }
    with open(os.path.join(OUTPUT_DIR, "cv5_test_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)

    print("\n===== TEST SUMMARY (CV 5-fold) =====")
    print("Best fold (by {}): fold {} | ckpt {} | loss={:.4f} dice={:.4f} acc={:.4f}".format(
        SELECT_METRIC,
        best_fold_rec["fold"],
        best_fold_rec["ckpt"],
        best_fold_rec["loss"],
        best_fold_rec["dice"],
        best_fold_rec["acc"],
    ))
    print("Ensemble: loss={:.4f} dice={:.4f} acc={:.4f}".format(
        cv_summary["ensemble"]["loss"],
        cv_summary["ensemble"]["dice"],
        cv_summary["ensemble"]["acc"],
    ))
    print("✓ Saved cv5_test_summary.json")

else:
    # ---- Fallback: single best model path (non-CV training) ----
    print("No fold checkpoints found; using single best_path:", best_path)
    ckpt = torch.load(best_path, map_location=DEVICE)
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()

    test_metrics = evaluate(model, test_loader)
    print("TEST METRICS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    print("✓ Saved test_metrics.json")


# In[6]:


# ========= Plots (single run OR 5-fold CV) =========
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _plot(series_dict, title, ylabel):
    plt.figure(figsize=(6,4))
    for label, series in series_dict.items():
        if series is None: 
            continue
        plt.plot(range(1, len(series)+1), series, label=label)
    plt.title(title); plt.xlabel("epoch"); plt.ylabel(ylabel); plt.legend(); plt.grid(True); plt.show()

def _load_cv_histories(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "history_fold*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    return files, dfs

def _stack_metric(dfs, key):
    # ensure same length across folds (in case of irregularities)
    min_len = min(len(df[key]) for df in dfs)
    arr = np.stack([df[key].values[:min_len] for df in dfs], axis=0)  # [folds, epochs]
    return arr

def plot_cv_histories(output_dir, train_key, val_key, title, ylabel):
    files, dfs = _load_cv_histories(output_dir)
    if not dfs:
        print(f"[plot_cv_histories] No per-fold histories found in {output_dir}.")
        return
    tr = _stack_metric(dfs, train_key)  # [F, T]
    va = _stack_metric(dfs, val_key)    # [F, T]
    epochs = np.arange(1, tr.shape[1]+1)

    # per-fold (faint)
    plt.figure(figsize=(7,4.5))
    for i in range(tr.shape[0]):
        plt.plot(epochs, tr[i], alpha=0.25)
    for i in range(va.shape[0]):
        plt.plot(epochs, va[i], alpha=0.25)

    # mean ± std
    tr_mean, tr_std = tr.mean(0), tr.std(0)
    va_mean, va_std = va.mean(0), va.std(0)

    plt.plot(epochs, tr_mean, label="Train (mean)", linewidth=2.5)
    plt.fill_between(epochs, tr_mean-tr_std, tr_mean+tr_std, alpha=0.2)

    plt.plot(epochs, va_mean, label="Val (mean)", linewidth=2.5)
    plt.fill_between(epochs, va_mean-va_std, va_mean+va_std, alpha=0.2)

    plt.title(title); plt.xlabel("epoch"); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ---- Single-run plots (if you trained a single split)
if 'history' in globals() and isinstance(history, dict) and len(history.get("train_loss", [])) > 0:
    _plot({"Train": history["train_loss"], "Val": history["val_loss"]}, "Loss", "BCE+Dice")
    _plot({"Train": history["train_dice"], "Val": history["val_dice"]}, "Dice", "Dice")
    _plot({"Train": history["train_acc"],  "Val": history["val_acc"]},  "Accuracy", "Accuracy")

# ---- CV plots (if you ran 5-fold and saved history_fold*.csv)
if 'OUTPUT_DIR' in globals():
    plot_cv_histories(OUTPUT_DIR, "train_loss", "val_loss", "Loss (CV 5-fold)", "BCE+Dice")
    plot_cv_histories(OUTPUT_DIR, "train_dice", "val_dice", "Dice (CV 5-fold)", "Dice")
    plot_cv_histories(OUTPUT_DIR, "train_acc",  "val_acc",  "Accuracy (CV 5-fold)", "Accuracy")


# In[7]:


# ========= Visualization: predictions vs ORIGINAL images (CV-aware) =========
import os, glob, json
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def _prepare_for_show(x_np):
    x = x_np.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    return np.clip(x, 0.0, 1.0)

def _list_fold_ckpts(output_dir):
    return sorted(glob.glob(os.path.join(output_dir, "unetpp_chase_best_fold*.pth")))

def _load_model_from_ckpt(ckpt_path):
    m = build_model()
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["state_dict"])
    m.to(DEVICE).eval()
    return m

@torch.no_grad()
def _visualize_core(model, loader, n=6, thr=0.5, title_prefix=""):
    """
    Shows: Original | Predicted mask | Ground-truth mask
    Uses ID2PATH_TEST (stem -> original image path) if available for true originals.
    """
    model.eval()
    shown = 0
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        ids  = batch["id"]
        logits = model(imgs)
        probs  = torch.sigmoid(logits)
        preds  = (probs > thr).float().cpu().numpy()
        gt     = batch["mask"].cpu().numpy()
        imgs_np = imgs.detach().cpu().numpy().transpose(0,2,3,1)

        for i in range(imgs.size(0)):
            if shown >= n:
                return
            # try to load original image from disk
            orig_path = None
            if 'ID2PATH_TEST' in globals():
                orig_path = ID2PATH_TEST.get(ids[i], None)

            if orig_path and os.path.exists(orig_path):
                orig = np.array(Image.open(orig_path).convert("RGB"))
            else:
                # fallback to the network input (resized)
                orig = (imgs_np[i] * 255.0).astype(np.uint8)

            orig_show = _prepare_for_show(orig)

            fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
            axs[0].imshow(orig_show);             axs[0].set_title(f"{title_prefix} | {ids[i]}"); axs[0].axis("off")
            axs[1].imshow(preds[i,0], cmap="gray"); axs[1].set_title("Pred");                      axs[1].axis("off")
            axs[2].imshow(gt[i,0],   cmap="gray");   axs[2].set_title("GT");                        axs[2].axis("off")
            plt.tight_layout(); plt.show()
            shown += 1

# ---------- Visualize: BEST fold if CV present, else single best_path ----------
@torch.no_grad()
def visualize_best(loader, n=6, thr=0.5, select_metric="dice"):
    """
    If CV folds found -> choose best fold by `select_metric` ('dice' or 'loss').
    Else -> use single-model best_path.
    """
    fold_ckpts = _list_fold_ckpts(OUTPUT_DIR)

    if len(fold_ckpts) == 0:
        # single-model path
        print("[Viz] No CV fold checkpoints found. Using single best_path.")
        ckpt = torch.load(best_path, map_location=DEVICE)
        model = build_model()
        model.load_state_dict(ckpt["state_dict"])
        model.to(DEVICE).eval()
        _visualize_core(model, loader, n=n, thr=thr, title_prefix="single")
        return

    # prefer saved test summary if available
    summary_path = os.path.join(OUTPUT_DIR, "cv5_test_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        per_fold = summary.get("per_fold", [])
        if select_metric.lower() == "loss":
            best = min(per_fold, key=lambda r: r.get("loss", float("inf")))
        else:
            best = max(per_fold, key=lambda r: r.get("dice", -float("inf")))
        best_fold = int(best["fold"])
        ckpt_path = os.path.join(OUTPUT_DIR, f"unetpp_chase_best_fold{best_fold}.pth")
        print(f"[Viz] Using BEST fold {best_fold} by '{select_metric}'.")
        model = _load_model_from_ckpt(ckpt_path)
        _visualize_core(model, loader, n=n, thr=thr, title_prefix=f"best fold {best_fold}")
        return

    # otherwise, score each fold on this loader and pick best on the fly
    print("[Viz] cv5_test_summary.json not found. Scoring each fold to choose best...")
    scores = []
    for i, ck in enumerate(fold_ckpts, 1):
        m = _load_model_from_ckpt(ck)
        met = evaluate(m, loader)  # uses your evaluate() (loss/dice/acc)
        scores.append({"fold": i, "ckpt": ck, **met})
        print(f"  Fold {i}: loss={met['loss']:.4f} dice={met['dice']:.4f} acc={met['acc']:.4f}")

    if select_metric.lower() == "loss":
        best = min(scores, key=lambda r: r["loss"])
    else:
        best = max(scores, key=lambda r: r["dice"])

    best_fold = int(best["fold"])
    print(f"[Viz] Using BEST fold {best_fold} by '{select_metric}'.")
    model = _load_model_from_ckpt(best["ckpt"])
    _visualize_core(model, loader, n=n, thr=thr, title_prefix=f"best fold {best_fold}")

# ---------- Visualize: ENSEMBLE across all folds ----------
@torch.no_grad()
def visualize_ensemble(loader, n=6, thr=0.5):
    ckpts = _list_fold_ckpts(OUTPUT_DIR)
    assert len(ckpts) > 0, "No CV fold checkpoints found."
    models = [_load_model_from_ckpt(ck) for ck in ckpts]
    print(f"[Viz] Ensemble of {len(models)} folds.")

    shown = 0
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        ids  = batch["id"]

        logits_sum = None
        for m in models:
            out = m(imgs)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        logits = logits_sum / len(models)
        probs  = torch.sigmoid(logits)
        preds  = (probs > thr).float().cpu().numpy()

        gt      = batch["mask"].cpu().numpy()
        imgs_np = imgs.detach().cpu().numpy().transpose(0,2,3,1)

        for i in range(imgs.size(0)):
            if shown >= n:
                return
            orig_path = None
            if 'ID2PATH_TEST' in globals():
                orig_path = ID2PATH_TEST.get(ids[i], None)

            if orig_path and os.path.exists(orig_path):
                orig = np.array(Image.open(orig_path).convert("RGB"))
            else:
                orig = (imgs_np[i] * 255.0).astype(np.uint8)
            orig_show = _prepare_for_show(orig)

            fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
            axs[0].imshow(orig_show);             axs[0].set_title(f"ensemble | {ids[i]}"); axs[0].axis("off")
            axs[1].imshow(preds[i,0], cmap="gray"); axs[1].set_title("Pred (ens)");         axs[1].axis("off")
            axs[2].imshow(gt[i,0],   cmap="gray");   axs[2].set_title("GT");                 axs[2].axis("off")
            plt.tight_layout(); plt.show()
            shown += 1

# ---- Call ONE (or both) of these:
# Best-by-metric visualization (defaults to Dice)
visualize_best(test_loader, n=6, thr=0.5, select_metric="dice")

# Or visualize an ensemble across all folds:
# visualize_ensemble(test_loader, n=6, thr=0.5)


# In[ ]:





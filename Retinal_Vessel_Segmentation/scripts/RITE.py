#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os, random, shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# =========================
# 0) CONFIG (edit ROOT if needed)
# =========================
ROOT = Path(r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images")  # has train/ and test/

# Folders
TRAIN_IMG_DIR  = ROOT / "train" / "img"
TRAIN_GT_DIR   = ROOT / "train" / "vessel"
TEST_IMG_DIR   = ROOT / "test"  / "img"
TEST_GT_DIR    = ROOT / "test"  / "vessel"   # optional

# Output
OUTPUT_DIR     = ROOT / "output_torch"
PRED_DIR       = OUTPUT_DIR / "preds"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PRED_DIR.mkdir(exist_ok=True, parents=True)

# Hyper-params (fixed as requested)
IMG_SIZE       = (512, 512)   # (H, W)
BATCH_SIZE     = 6
EPOCHS         = 60
LEARNING_RATE  = 1e-4
ACC_THR        = 0.5
N_FOLDS        = 5
SEED           = 42
PATIENCE       = 10           # early stopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# Sanity checks
for p in [TRAIN_IMG_DIR, TRAIN_GT_DIR, TEST_IMG_DIR]:
    assert p.exists(), f"Missing: {p}"
if not TEST_GT_DIR.exists():
    print("[INFO] Test GT not found; test metrics will be skipped (prediction-only).")

# Repro
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# =========================
# 1) Pairing helpers
# =========================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}
def is_img(p: Path): return p.is_file() and p.suffix.lower() in IMG_EXTS

MANUAL_SUFFIXES = ["_manual1", "-manual1", "_vessel", "_vessels", "-vessel", "-vessels"]
IMAGE_SUFFIXES  = ["_test", "-test", "_train", "-train", "_training", "-training"]

def strip_any_suffix(stem: str, suffixes):
    s = stem.lower()
    for suf in suffixes:
        if s.endswith(suf):
            return stem[:len(stem)-len(suf)]
    return stem

def index_by_stem(paths, extra_suffixes):
    idx = defaultdict(list)
    for p in paths:
        st = p.stem
        idx[st].append(p)
        idx[st.lower()].append(p)
        base = strip_any_suffix(st, extra_suffixes)
        idx[base].append(p)
        idx[base.lower()].append(p)
    return idx

def try_lookup(idx, key): return list({*idx.get(key, []), *idx.get(key.lower(), [])})

def collect_pairs(img_dir: Path, gt_dir: Path | None):
    imgs = sorted([p for p in img_dir.rglob("*") if is_img(p)])
    gts  = sorted([p for p in gt_dir.rglob("*") if is_img(p)]) if gt_dir and gt_dir.exists() else []
    man_idx = index_by_stem(gts, MANUAL_SUFFIXES) if gts else {}

    pairs, missing = [], []
    for ip in imgs:
        img_stem = ip.stem
        img_base = strip_any_suffix(img_stem, IMAGE_SUFFIXES)

        gt_path = None
        if man_idx:
            cand = []
            for k in (img_stem, img_base):
                cand += try_lookup(man_idx, k)
            if not cand:
                for suf in MANUAL_SUFFIXES:
                    cand += try_lookup(man_idx, img_stem + suf)
                    cand += try_lookup(man_idx, img_base + suf)
            cand = list(set(cand))
            if cand:
                gt_path = sorted(cand)[0]
            else:
                missing.append(ip.name)
                continue  # need GT for training/eval

        pairs.append({"img": ip.resolve(), "gt": gt_path.resolve() if gt_path else None})
    return pairs, missing

# =========================
# 2) Robust PIL readers (handle 8/16-bit TIFFs cleanly)
# =========================
def pil_read_rgb(path: Path):
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA",): im = im.convert("RGB")
        elif im.mode in ("L",):  im = im.convert("RGB")
        elif im.mode in ("I;16","I"):
            arr = np.array(im, dtype=np.float32); mx = arr.max() if arr.max()>0 else 1.0
            arr = (arr/mx*255.0).clip(0,255).astype(np.uint8)
            im = Image.fromarray(arr).convert("RGB")
        elif im.mode not in ("RGB",): im = im.convert("RGB")
        return np.array(im)

def pil_read_mask(path: Path):
    with Image.open(path) as im:
        im.load()
        if im.mode not in ("L","1"):
            arr = np.array(im)
            if arr.ndim == 3: arr = arr[...,0]
        else:
            arr = np.array(im)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32); mx = arr.max() if arr.max()>0 else 1.0
            arr = (arr/mx*255.0).clip(0,255).astype(np.uint8)
        arr = (arr > 0).astype(np.uint8)
        return arr

def validate_records(records, kind="train", max_report=10):
    ok, bad = [], []
    for r in records:
        try:
            _ = pil_read_rgb(r["img"])
            if r["gt"] is not None:
                m = pil_read_mask(r["gt"])
                if m.ndim != 2: raise RuntimeError("Mask not 2D.")
        except (FileNotFoundError, UnidentifiedImageError, OSError, RuntimeError) as e:
            bad.append((r, str(e)))
        else:
            ok.append(r)
    if bad:
        print(f"[WARN] {kind}: {len(bad)} bad record(s) skipped.")
        for i,(r,msg) in enumerate(bad[:max_report]):
            print(f"  - {r['img'].name} | GT: {r['gt'].name if r['gt'] else None} | {msg}")
        if len(bad) > max_report:
            print(f"  ... and {len(bad)-max_report} more")
    print(f"[INFO] {kind}: {len(ok)} valid record(s).")
    return ok


# In[4]:


train_pool, miss_tr = collect_pairs(TRAIN_IMG_DIR, TRAIN_GT_DIR)
print(f"[PAIR] TRAIN paired: {len(train_pool)} (missing GT {len(miss_tr)})")
train_pool = validate_records(train_pool, "train")
assert len(train_pool) >= N_FOLDS, f"Need at least {N_FOLDS} train samples, got {len(train_pool)}."

if TEST_GT_DIR.exists():
    test_pairs, miss_te = collect_pairs(TEST_IMG_DIR, TEST_GT_DIR)
    print(f"[PAIR] TEST paired:  {len(test_pairs)} (missing GT {len(miss_te)})")
    test_pairs = validate_records(test_pairs, "test")
else:
    test_pairs = [{"img": p.resolve(), "gt": None} for p in sorted([p for p in TEST_IMG_DIR.rglob("*") if is_img(p)])]
    print(f"[PAIR] TEST prediction-only images: {len(test_pairs)}")


# In[5]:


# =========================
train_tf = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.2),
    A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ToTensorV2()
])
val_tf = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
    ToTensorV2()
])

class VesselDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.tf = transform
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        rec = self.records[i]
        img = pil_read_rgb(rec["img"])
        if rec["gt"] is None:
            aug = self.tf(image=img)
            img_t = aug["image"]
            msk_t = torch.zeros(1, IMG_SIZE[0], IMG_SIZE[1], dtype=torch.float32)
            return img_t, msk_t
        mask = pil_read_mask(rec["gt"])  # HxW {0,1}
        aug = self.tf(image=img, mask=mask)
        img_t = aug["image"]
        m = aug["mask"]
        if isinstance(m, torch.Tensor):
            msk_t = m.float()
            if msk_t.ndim == 2: msk_t = msk_t.unsqueeze(0)
            elif msk_t.ndim == 3 and msk_t.shape[0] != 1: msk_t = msk_t.permute(2,0,1).float()
        else:
            msk_t = torch.from_numpy(m).float().unsqueeze(0)
        msk_t = (msk_t > 0).float()
        return img_t, msk_t

# =========================
# 5) Dataloaders
# =========================
NUM_WORKERS = 0
PIN_MEMORY  = True if DEVICE.type == "cuda" else False

def make_loader(recs, tfm, shuffle):
    ds = VesselDataset(recs, tfm)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)

# =========================
# 6) Model / loss / metrics
# =========================
def build_model():
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

bce_logits = nn.BCEWithLogitsLoss()

@torch.no_grad()
def dice_coef(pred_prob, target, eps=1e-6):
    pred = (pred_prob >= ACC_THR).float()
    target = target.float()
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    return (2*inter + eps) / (denom + eps)

@torch.no_grad()
def bin_accuracy(pred_prob, target):
    pred = (pred_prob >= ACC_THR).float()
    target = target.float()
    return (pred == target).float().mean()


# In[7]:


# add once near your imports
import pandas as pd

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = total_dice = total_acc = 0.0
    n = 0
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE); masks = masks.to(DEVICE)
        logits = model(imgs)
        loss = bce_logits(logits, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(logits); bs = imgs.size(0)
            total_loss += loss.item() * bs
            # per-sample aggregation (your original style)
            for i in range(bs):
                total_dice += dice_coef(probs[i], masks[i]).item()
                total_acc  += bin_accuracy(probs[i], masks[i]).item()
            n += bs
    return total_loss/n, total_dice/n, total_acc/n

@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    total_loss = total_dice = total_acc = 0.0
    n = 0
    for imgs, masks in loader:
        imgs  = imgs.to(DEVICE); masks = masks.to(DEVICE)
        logits = model(imgs); loss = bce_logits(logits, masks)
        probs = torch.sigmoid(logits); bs = imgs.size(0)
        total_loss += loss.item() * bs
        for i in range(bs):
            total_dice += dice_coef(probs[i], masks[i]).item()
            total_acc  += bin_accuracy(probs[i], masks[i]).item()
        n += bs
    return total_loss/n, total_dice/n, total_acc/n

def plot_history(hist, out_dir: Path, prefix=""):
    tr_loss = [h["tl"] for h in hist]; va_loss = [h["vl"] for h in hist]
    tr_dice = [h["td"] for h in hist]; va_dice = [h["vd"] for h in hist]
    tr_acc  = [h["ta"] for h in hist]; va_acc  = [h["va"] for h in hist]
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(tr_loss, label="train"); plt.plot(va_loss, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{prefix}Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}history_loss.png", dpi=150); plt.show()
    # Dice
    plt.figure(figsize=(6,4))
    plt.plot(tr_dice, label="train"); plt.plot(va_dice, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.title(f"{prefix}Dice"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}history_dice.png", dpi=150); plt.show()
    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(tr_acc, label="train"); plt.plot(va_acc, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{prefix}Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}history_accuracy.png", dpi=150); plt.show()

# =========================
# 8) 5-fold CV on train_pool
# =========================
indices = np.arange(len(train_pool))
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

best_overall = {"fold": None, "val_dice": -1.0, "ckpt": None}
fold_summaries = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(indices), start=1):
    tr_recs = [train_pool[i] for i in tr_idx]
    va_recs = [train_pool[i] for i in va_idx]
    print(f"\n========== Fold {fold}/{N_FOLDS} ==========")
    print(f"Fold sizes → train {len(tr_recs)} | val {len(va_recs)}")

    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True, parents=True)

    train_loader = make_loader(tr_recs, train_tf, shuffle=True)
    val_loader   = make_loader(va_recs, val_tf,   shuffle=False)

    # quick batch sanity (optional)
    imgs_chk, masks_chk = next(iter(train_loader))
    assert imgs_chk.ndim == 4 and masks_chk.ndim == 4 and masks_chk.shape[1] == 1, \
        f"Bad batch shapes: imgs {imgs_chk.shape}, masks {masks_chk.shape}"

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_dice = -1.0
    wait = 0
    ckpt_path = fold_dir / "best_unetpp_fold.pth"

    for epoch in range(1, EPOCHS+1):
        tl, td, ta = train_one_epoch(model, train_loader, optimizer)
        vl, vd, va = eval_one_epoch(model, val_loader)

        # track LR too (if you use schedulers later this will reflect changes)
        cur_lr = optimizer.param_groups[0]["lr"]

        history.append({"epoch": epoch,
                        "tl": tl, "td": td, "ta": ta,
                        "vl": vl, "vd": vd, "va": va,
                        "lr": cur_lr})

        print(f"[Fold {fold:02d}] Epoch {epoch:03d}/{EPOCHS}  "
              f"loss: {tl:.4f}/{vl:.4f}  dice: {td:.4f}/{vd:.4f}  acc: {ta:.4f}/{va:.4f}")

        if vd > best_val_dice:
            best_val_dice = vd
            torch.save(model.state_dict(), ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val Dice {best_val_dice:.4f}")
                break

    # ---- save per-fold history CSV ----
    hist_df = pd.DataFrame(history,
                           columns=["epoch","tl","td","ta","vl","vd","va","lr"])
    hist_csv = fold_dir / f"history_fold{fold}.csv"
    hist_df.to_csv(hist_csv, index=False)
    print(f"Saved history → {hist_csv}")

    # plots
    plot_history(history, fold_dir, prefix=f"fold{fold}_")

    fold_summaries.append({"fold": fold,
                           "best_val_dice": float(best_val_dice),
                           "ckpt": str(ckpt_path)})

    if best_val_dice > best_overall["val_dice"]:
        best_overall = {"fold": fold,
                        "val_dice": float(best_val_dice),
                        "ckpt": str(ckpt_path)}

print("\n=== CV summary (best val Dice per fold) ===")
for fs in fold_summaries:
    print(f"Fold {fs['fold']}: {fs['best_val_dice']:.4f}")
print(f"BEST FOLD: {best_overall['fold']}  (val Dice = {best_overall['val_dice']:.4f})")

BEST_MODEL_PATH = OUTPUT_DIR / "best_cv_unetpp.pth"
shutil.copyfile(best_overall["ckpt"], BEST_MODEL_PATH)
print(f"Saved best fold weights → {BEST_MODEL_PATH}")


# In[ ]:


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss=total_dice=total_acc=n=0
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
        logits = model(imgs)
        loss = bce_logits(logits, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(logits); bs = imgs.size(0)
            total_loss += loss.item()*bs
            for i in range(bs):
                total_dice += dice_coef(probs[i], masks[i]).item()
                total_acc  += bin_accuracy(probs[i], masks[i]).item()
            n += bs
    return total_loss/n, total_dice/n, total_acc/n

@torch.no_grad()
def eval_one_epoch(model, loader):
    model.eval()
    total_loss=total_dice=total_acc=n=0
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
        logits = model(imgs); loss = bce_logits(logits, masks)
        probs = torch.sigmoid(logits); bs = imgs.size(0)
        total_loss += loss.item()*bs
        for i in range(bs):
            total_dice += dice_coef(probs[i], masks[i]).item()
            total_acc  += bin_accuracy(probs[i], masks[i]).item()
        n += bs
    return total_loss/n, total_dice/n, total_acc/n

def plot_history(hist, out_dir: Path, prefix=""):
    tr_loss = [h["tl"] for h in hist]; va_loss = [h["vl"] for h in hist]
    tr_dice = [h["td"] for h in hist]; va_dice = [h["vd"] for h in hist]
    tr_acc  = [h["ta"] for h in hist]; va_acc  = [h["va"] for h in hist]
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(tr_loss, label="train"); plt.plot(va_loss, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{prefix}Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}history_loss.png", dpi=150); plt.show()
    # Dice
    plt.figure(figsize=(6,4))
    plt.plot(tr_dice, label="train"); plt.plot(va_dice, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Dice"); plt.title(f"{prefix}Dice"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}history_dice.png", dpi=150); plt.show()
    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(tr_acc, label="train"); plt.plot(va_acc, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{prefix}Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}history_accuracy.png", dpi=150); plt.show()

# =========================
# 8) 5-fold CV on train_pool
# =========================
indices = np.arange(len(train_pool))
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

best_overall = {"fold": None, "val_dice": -1.0, "ckpt": None}
fold_summaries = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(indices), start=1):
    tr_recs = [train_pool[i] for i in tr_idx]
    va_recs = [train_pool[i] for i in va_idx]
    print(f"\n========== Fold {fold}/{N_FOLDS} ==========")
    print(f"Fold sizes → train {len(tr_recs)} | val {len(va_recs)}")  # prints validation length per fold

    fold_dir = OUTPUT_DIR / f"fold_{fold}"
    fold_dir.mkdir(exist_ok=True, parents=True)

    train_loader = make_loader(tr_recs, train_tf, shuffle=True)
    val_loader   = make_loader(va_recs, val_tf,   shuffle=False)

    # quick batch sanity (optional)
    imgs_chk, masks_chk = next(iter(train_loader))
    assert imgs_chk.ndim == 4 and masks_chk.ndim == 4 and masks_chk.shape[1] == 1, \
        f"Bad batch shapes: imgs {imgs_chk.shape}, masks {masks_chk.shape}"

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_dice = -1.0
    wait = 0
    ckpt_path = fold_dir / "best_unetpp_fold.pth"

    for epoch in range(1, EPOCHS+1):
        tl, td, ta = train_one_epoch(model, train_loader, optimizer)
        vl, vd, va = eval_one_epoch(model, val_loader)
        history.append({"tl":tl, "td":td, "ta":ta, "vl":vl, "vd":vd, "va":va})

        print(f"[Fold {fold:02d}] Epoch {epoch:03d}/{EPOCHS}  "
              f"loss: {tl:.4f}/{vl:.4f}  dice: {td:.4f}/{vd:.4f}  acc: {ta:.4f}/{va:.4f}")

        if vd > best_val_dice:
            best_val_dice = vd
            torch.save(model.state_dict(), ckpt_path)
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val Dice {best_val_dice:.4f}")
                break

    plot_history(history, fold_dir, prefix=f"fold{fold}_")
    fold_summaries.append({"fold": fold, "best_val_dice": float(best_val_dice), "ckpt": str(ckpt_path)})

    if best_val_dice > best_overall["val_dice"]:
        best_overall = {"fold": fold, "val_dice": float(best_val_dice), "ckpt": str(ckpt_path)}

print("\n=== CV summary (best val Dice per fold) ===")
for fs in fold_summaries:
    print(f"Fold {fs['fold']}: {fs['best_val_dice']:.4f}")
print(f"BEST FOLD: {best_overall['fold']}  (val Dice = {best_overall['val_dice']:.4f})")

BEST_MODEL_PATH = OUTPUT_DIR / "best_cv_unetpp.pth"
shutil.copyfile(best_overall["ckpt"], BEST_MODEL_PATH)
print(f"Saved best fold weights → {BEST_MODEL_PATH}")


# In[10]:


# === Visualize N-fold histories (robust; works with tl/vl/td/vd/ta/va) ===
import os, glob, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIG — set to the parent directory that contains fold_1, fold_2, ...
# ---------------------------------------------------------------------
OUTPUT_DIR = r"C:\Users\SC\Documents\Data and Code\Rite benchmarks\images\output_torch"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Mean±Std strategy when folds have different #epochs: "pad" or "truncate"
MEAN_STD_STRATEGY = "pad"   # "pad" (NaN pad) | "truncate" (cut to shortest)

# ---------------------------------------------------------------------
# Column normalization: map whatever you logged to a standard set
# ---------------------------------------------------------------------
STD_KEYS = {
    "train_loss": ["train_loss", "tl", "loss_train"],
    "val_loss":   ["val_loss",   "vl", "loss_val"],
    "train_dice": ["train_dice", "td", "dice_train"],
    "val_dice":   ["val_dice",   "vd", "dice_val"],
    "train_acc":  ["train_acc",  "ta", "acc_train", "accuracy_train"],
    "val_acc":    ["val_acc",    "va", "acc_val",   "accuracy_val"],
    "lr":         ["lr", "learning_rate"]
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names and coerce numeric."""
    colmap = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for std, variants in STD_KEYS.items():
        for v in variants:
            if v.lower() in lower_cols:
                colmap[lower_cols[v.lower()]] = std
                break
    if colmap:
        df = df.rename(columns=colmap)

    # Ensure epoch exists and is numeric
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df) + 1))
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    # Force numeric metrics; bad cells → NaN
    for k in ["train_loss","val_loss","train_dice","val_dice","train_acc","val_acc","lr"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    return df

# ---------------------------------------------------------------------
# Find and read histories
# ---------------------------------------------------------------------
FOLD_RE = re.compile(r"fold[_\- ]?(\d+)", re.IGNORECASE)

def find_history_files(output_dir):
    """
    Look for history files in several common layouts:
      <out>/history*.csv
      <out>/fold_*/history*.csv
      <out>/fold_*/history*        (even if extension is missing)
      <out>/fold_*/*.csv
    """
    patterns = [
        os.path.join(output_dir, "history*.csv"),
        os.path.join(output_dir, "fold_*", "history*.csv"),
        os.path.join(output_dir, "fold-*", "history*.csv"),
        os.path.join(output_dir, "fold*",  "history*.csv"),
        os.path.join(output_dir, "fold_*", "history*"),
        os.path.join(output_dir, "fold-*", "history*"),
        os.path.join(output_dir, "fold*",  "history*"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))

def infer_fold_index(path):
    """Return fold index from filename/parent, else the basename as a key."""
    base   = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    for token in (base, parent):
        m = FOLD_RE.search(token)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    digits = re.findall(r"\d+", base)
    if digits:
        return int(digits[-1])
    return base

def load_history(path):
    """Read CSV (or sniff separator), normalize columns."""
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            # sep=None with engine='python' lets Pandas sniff commas/semicolons/tabs
            df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        print(f"[error] failed to read {path}: {e}")
        return None
    return normalize_columns(df)

# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------
def best_epoch(df, by="val_loss", mode="min"):
    if df is None or by not in df.columns:
        return None, None
    idx = int(df[by].idxmin() if mode == "min" else df[by].idxmax())
    return idx + 1, df.iloc[idx].to_dict()

def _stack(mats, strategy="pad"):
    """Stack 1D arrays with different lengths."""
    if not mats:
        return None, None
    if strategy == "truncate":
        m = min(len(a) for a in mats)
        return np.vstack([a[:m] for a in mats]), np.arange(1, m + 1)
    # pad by default
    L = max(len(a) for a in mats)
    out = np.full((len(mats), L), np.nan, dtype=float)
    for i, a in enumerate(mats):
        out[i, :len(a)] = a
    return out, np.arange(1, L + 1)

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def overlay_plot(histories, key_train, key_val, title, ylabel, savepath=None):
    plt.figure(figsize=(8, 5))
    have_any = False
    for k, df in sorted(histories.items(), key=lambda x: str(x[0])):
        if df is None:
            continue
        if key_train in df.columns:
            plt.plot(df["epoch"], df[key_train], alpha=0.65, lw=1.6, label=f"Fold {k} Train")
            have_any = True
        if key_val in df.columns:
            plt.plot(df["epoch"], df[key_val], alpha=0.95, lw=2.0, ls="--", label=f"Fold {k} Val")
            have_any = True
    if not have_any:
        print(f"[warn] No data for {title}")
        plt.close(); return
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200)
    plt.show(); plt.close()

def mean_std_plot(histories, key_train, key_val, title, ylabel, savepath=None):
    mats_tr, mats_va = [], []
    for _, df in histories.items():
        if df is None: continue
        if key_train in df.columns: mats_tr.append(df[key_train].values)
        if key_val   in df.columns: mats_va.append(df[key_val].values)
    if not mats_tr or not mats_va:
        print(f"[warn] no histories found for mean±std: {key_train}/{key_val}")
        return
    tr, x = _stack(mats_tr, strategy=MEAN_STD_STRATEGY)
    va, _ = _stack(mats_va, strategy=MEAN_STD_STRATEGY)
    if tr is None or va is None:
        print(f"[warn] could not stack arrays for {title}")
        return
    tr_mean, tr_std = np.nanmean(tr, axis=0), np.nanstd(tr, axis=0)
    va_mean, va_std = np.nanmean(va, axis=0), np.nanstd(va, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(x, tr_mean, lw=2.4, label="Train (mean)")
    plt.fill_between(x, tr_mean - tr_std, tr_mean + tr_std, alpha=0.18, label="Train (±1σ)")
    plt.plot(x, va_mean, lw=2.4, ls="--", label="Val (mean)")
    plt.fill_between(x, va_mean - va_std, va_mean + va_std, alpha=0.18, label="Val (±1σ)")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200)
    plt.show(); plt.close()

def plot_lr(histories, savepath=None):
    plt.figure(figsize=(7, 4))
    have_any = False
    for k, df in sorted(histories.items(), key=lambda x: str(x[0])):
        if df is None or "lr" not in df.columns: 
            continue
        plt.plot(df["epoch"], df["lr"], label=f"Fold {k}")
        have_any = True
    if not have_any:
        print("[info] no 'lr' columns found.")
        plt.close(); return
    plt.title("Learning Rate per Fold"); plt.xlabel("Epoch"); plt.ylabel("LR")
    plt.grid(True, alpha=0.25); plt.legend(); plt.tight_layout()
    if savepath: plt.savefig(savepath, dpi=200)
    plt.show(); plt.close()

# ---------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------
files = find_history_files(OUTPUT_DIR)
if not files:
    print(f"[error] No history files found under: {OUTPUT_DIR}")
else:
    print("Found history files:")
    for f in files: print(" •", f)

histories = {}
for fpath in files:
    key = infer_fold_index(fpath)
    df  = load_history(fpath)
    histories[key] = df

# Quick sanity print
for k, df in sorted(histories.items(), key=lambda x: str(x[0])):
    if df is None:
        print(f"Fold {k}: <none>")
    else:
        print(f"Fold {k}: columns -> {list(df.columns)} (len={len(df)})")

# ---------------------------------------------------------------------
# Per-fold “best epoch” table (by min val_loss and max val_dice)
# ---------------------------------------------------------------------
rows = []
for f, df in sorted(histories.items(), key=lambda x: str(x[0])):
    if df is None: 
        continue
    e_min, row_min = best_epoch(df, by="val_loss", mode="min")
    e_max, row_max = best_epoch(df, by="val_dice", mode="max")
    rows.append({
        "fold": f,
        "best_epoch_by_val_loss": e_min,
        "val_loss(best)": None if row_min is None else round(row_min.get("val_loss", np.nan), 4),
        "val_dice(at_best_loss)": None if row_min is None else round(row_min.get("val_dice", np.nan), 4),
        "best_epoch_by_val_dice": e_max,
        "val_dice(best)": None if row_max is None else round(row_max.get("val_dice", np.nan), 4),
        "val_acc(at_best_dice)": None if row_max is None else round(row_max.get("val_acc", np.nan), 4),
    })

if rows:
    best_table = pd.DataFrame(rows).sort_values("fold")
    print("\nPer-fold best epochs:")
    display(best_table)
else:
    print("\n[info] No valid histories to form a best-epoch table.")

# ---------------------------------------------------------------------
# Per-fold 3-panel plots
# ---------------------------------------------------------------------
for f, df in sorted(histories.items(), key=lambda x: str(x[0])):
    if df is None:
        print(f"[warn] no data for fold {f}")
        continue

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Loss
    if "train_loss" in df.columns and "val_loss" in df.columns:
        axs[0].plot(df["epoch"], df["train_loss"], label="Train")
        axs[0].plot(df["epoch"], df["val_loss"],   label="Val", ls="--")
    else:
        axs[0].text(0.5, 0.5, "loss not found", ha="center")
    axs[0].set_title(f"Fold {f} — Loss"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss")
    axs[0].grid(True); axs[0].legend()

    # Dice
    if "train_dice" in df.columns and "val_dice" in df.columns:
        axs[1].plot(df["epoch"], df["train_dice"], label="Train")
        axs[1].plot(df["epoch"], df["val_dice"],   label="Val", ls="--")
    else:
        axs[1].text(0.5, 0.5, "dice not found", ha="center")
    axs[1].set_title(f"Fold {f} — Dice"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Dice")
    axs[1].grid(True); axs[1].legend()

    # Accuracy
    if "train_acc" in df.columns and "val_acc" in df.columns:
        axs[2].plot(df["epoch"], df["train_acc"], label="Train")
        axs[2].plot(df["epoch"], df["val_acc"],   label="Val", ls="--")
    else:
        axs[2].text(0.5, 0.5, "acc not found", ha="center")
    axs[2].set_title(f"Fold {f} — Accuracy"); axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("Accuracy")
    axs[2].grid(True); axs[2].legend()

    plt.tight_layout()
    out_png = os.path.join(FIG_DIR, f"fold{f}_summary.png")
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")
    plt.show(); plt.close()

# ---------------------------------------------------------------------
# Overlays & Mean±Std
# ---------------------------------------------------------------------
overlay_plot(histories, "train_loss", "val_loss", "Loss — all folds", "Loss",
             savepath=os.path.join(FIG_DIR, "loss_all_folds.png"))
overlay_plot(histories, "train_dice", "val_dice", "Dice — all folds", "Dice",
             savepath=os.path.join(FIG_DIR, "dice_all_folds.png"))
overlay_plot(histories, "train_acc",  "val_acc",  "Accuracy — all folds", "Accuracy",
             savepath=os.path.join(FIG_DIR, "acc_all_folds.png"))

mean_std_plot(histories, "train_loss", "val_loss",
              f"Loss — mean ± std across folds ({MEAN_STD_STRATEGY})", "Loss",
              savepath=os.path.join(FIG_DIR, "loss_mean_std.png"))
mean_std_plot(histories, "train_dice", "val_dice",
              f"Dice — mean ± std across folds ({MEAN_STD_STRATEGY})", "Dice",
              savepath=os.path.join(FIG_DIR, "dice_mean_std.png"))
mean_std_plot(histories, "train_acc",  "val_acc",
              f"Accuracy — mean ± std across folds ({MEAN_STD_STRATEGY})", "Accuracy",
              savepath=os.path.join(FIG_DIR, "acc_mean_std.png"))

# LR schedule, if your CSVs contain an 'lr' column
plot_lr(histories, savepath=os.path.join(FIG_DIR, "lr_all_folds.png"))

print("\nAll done. Figures saved in:", FIG_DIR)


# In[11]:


def save_pred_png(img_path: Path, prob_map: np.ndarray, out_dir: Path):
    pred_bin = (prob_map >= ACC_THR).astype(np.uint8) * 255
    Image.fromarray(pred_bin).save(out_dir / f"{img_path.stem}_pred.png")

# reload best model
best_model = build_model().to(DEVICE)
best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
best_model.eval()

if test_pairs and test_pairs[0]["gt"] is not None:
    test_loader = make_loader(test_pairs, val_tf, shuffle=False)
    with torch.no_grad():
        tot_loss=tot_dice=tot_acc=n=0
        bce = nn.BCEWithLogitsLoss()
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
            logits = best_model(imgs)
            loss = bce(logits, masks)
            probs = torch.sigmoid(logits)
            bs = imgs.size(0)
            tot_loss += loss.item()*bs
            for i in range(bs):
                tot_dice += dice_coef(probs[i], masks[i]).item()
                tot_acc  += bin_accuracy(probs[i], masks[i]).item()
            n += bs
    print(f"\nTEST metrics  loss={tot_loss/n:.4f}  dice={tot_dice/n:.4f}  acc={tot_acc/n:.4f}")
else:
    print("\n[INFO] No GT for test; skipping test metrics (prediction-only).")

# predictions (all test images)
print("Writing test predictions from BEST CV model...")
for rec in test_pairs:
    img = pil_read_rgb(rec["img"])
    aug = val_tf(image=img)
    x = aug["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(best_model(x))[0,0].cpu().numpy()
    save_pred_png(rec["img"], prob, PRED_DIR)

print(f"Predictions saved to: {PRED_DIR}")


# In[12]:


def show_triptychs(sample_recs, model, k=3, title="Test: image | GT | prediction"):
    sample = sample_recs if len(sample_recs) <= k else random.sample(sample_recs, k)
    plt.figure(figsize=(12, 4*len(sample)))
    for i, rec in enumerate(sample):
        img = pil_read_rgb(rec['img'])
        aug = val_tf(image=img)
        x = aug["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
        plt.subplot(len(sample), 3, 3*i+1); plt.imshow(img); plt.axis("off"); plt.title(rec['img'].name)
        if rec["gt"] is not None:
            gt = pil_read_mask(rec["gt"]).astype(bool)
            gt = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((IMG_SIZE[1], IMG_SIZE[0]), Image.NEAREST)) > 0
            plt.subplot(len(sample), 3, 3*i+2); plt.imshow(gt, cmap="gray"); plt.axis("off"); plt.title("GT")
        else:
            plt.subplot(len(sample), 3, 3*i+2); plt.imshow(np.zeros(IMG_SIZE), cmap="gray"); plt.axis("off"); plt.title("GT (none)")
        plt.subplot(len(sample), 3, 3*i+3); plt.imshow(prob >= ACC_THR, cmap="gray"); plt.axis("off"); plt.title("Prediction")
    plt.suptitle(title); plt.tight_layout(); plt.show()

show_triptychs(test_pairs, best_model, k=3, title="Test: image | GT | UNet++ prediction (best fold)")


# In[ ]:





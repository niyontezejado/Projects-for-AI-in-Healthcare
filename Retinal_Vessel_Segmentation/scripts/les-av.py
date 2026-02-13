#!/usr/bin/env python
# coding: utf-8

# In[7]:


# ==== HRF pairing: images ↔ manual1 (GT), optional FoV mask ====
import os, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
# ---- change only ROOT if your path differs
ROOT = Path(r"C:\Users\SC\Documents\Data and Code\LES-AV\LES-AV")
IMAGES_DIR  = ROOT / "images"
MANUAL1_DIR = ROOT / "vessel-segmentations"
FOV_DIR     = ROOT / "masks"       # optional

assert IMAGES_DIR.exists(),  f"Missing: {IMAGES_DIR}"
assert MANUAL1_DIR.exists(), f"Missing: {MANUAL1_DIR}"
if not FOV_DIR.exists():
    print("[WARN] FoV folder not found; continuing without FoV masks.")
    
SEED = 42
random.seed(SEED)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ppm"}
def is_img(p: Path): return p.is_file() and p.suffix.lower() in IMG_EXTS

# HRF manual suffixes we may encounter
MANUAL_SUFFIXES = ["_manual1", "-manual1", "_vessel", "_vessels", "-vessel", "-vessels"]
MASK_SUFFIXES   = ["_mask", "-mask", "_fov", "-fov"]

def strip_any_suffix(stem: str, suffixes):
    s = stem.lower()
    for suf in suffixes:
        if s.endswith(suf):
            return stem[:len(stem)-len(suf)]
    return stem

# collect files
imgs  = sorted([p for p in IMAGES_DIR.rglob("*")  if is_img(p)])
mans  = sorted([p for p in MANUAL1_DIR.rglob("*") if is_img(p)])
fovs  = sorted([p for p in FOV_DIR.rglob("*")     if is_img(p)]) if FOV_DIR.exists() else []

# index manual1 and fov by several keys
def index_by_stem(paths, extra_suffixes):
    idx = defaultdict(list)
    for p in paths:
        st = p.stem
        idx[st].append(p)                # exact
        idx[st.lower()].append(p)        # lower
        base = strip_any_suffix(st, extra_suffixes)
        idx[base].append(p)
        idx[base.lower()].append(p)
    return idx

man_idx = index_by_stem(mans, MANUAL_SUFFIXES)
fov_idx = index_by_stem(fovs, MASK_SUFFIXES) if fovs else {}

pairs, missing_gt = [], []
for ip in imgs:
    key1 = ip.stem; key2 = ip.stem.lower()
    cand = list({*man_idx.get(key1, []), *man_idx.get(key2, [])})
    if not cand:
        # last try: attach known suffixes and re-check
        for suf in MANUAL_SUFFIXES:
            cand += man_idx.get(ip.stem + suf, [])
    cand = list(set(cand))
    if not cand:
        missing_gt.append(ip.name)
        continue
    # choose lexicographically stable (usually one)
    gt_path = sorted(cand)[0]
    # fov optional
    fov_path = None
    if fov_idx:
        c2 = list({*fov_idx.get(key1, []), *fov_idx.get(key2, [])})
        if not c2:
            for suf in MASK_SUFFIXES:
                c2 += fov_idx.get(ip.stem + suf, [])
        c2 = list(set(c2))
        if c2:
            fov_path = sorted(c2)[0]
    pairs.append({"img": ip, "gt": gt_path, "fov": fov_path})

print(f"Found {len(imgs)} images | Paired {len(pairs)} | Unpaired (no GT) {len(missing_gt)}")
if missing_gt: print("Examples:", missing_gt[:5])

# 70/15/15 split
random.Random(SEED).shuffle(pairs)
n = len(pairs)
n_tr = int(round(n*0.70))
n_va = int(round(n*0.15))
n_te = n - n_tr - n_va

train_recs = pairs[:n_tr]
val_recs   = pairs[n_tr:n_tr+n_va]
test_recs  = pairs[n_tr+n_va:]

print(f"SPLIT -> train {len(train_recs)} | val {len(val_recs)} | test {len(test_recs)}")

# map for true-original visualization later
ID2PATH_TEST = {r["img"].stem: str(r["img"]) for r in test_recs}


# In[8]:


# ==== sanity check: 3 samples with overlay ====
import matplotlib.pyplot as plt

def _overlay_mask(img, msk, color=(255,0,0), alpha=0.45):
    v = img.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32)
    idx = msk.astype(bool)
    v[idx] = (1-alpha)*v[idx] + alpha*col
    return v.clip(0,255).astype(np.uint8)

def show_samples(records, n=3, seed=42, title="train"):
    picks = random.Random(seed).sample(records, k=min(n, len(records)))
    for r in picks:
        img = np.array(Image.open(r["img"]).convert("RGB"))
        gt  = np.array(Image.open(r["gt"]).convert("L"))
        gt  = (gt > 0).astype(np.uint8)
        ov  = _overlay_mask(img, gt)
        fig, axs = plt.subplots(1,3, figsize=(11,3.5))
        axs[0].imshow(img); axs[0].set_title(f"{title} | {r['img'].stem}"); axs[0].axis("off")
        axs[1].imshow(gt,cmap="gray"); axs[1].set_title("GT (manual1)"); axs[1].axis("off")
        axs[2].imshow(ov); axs[2].set_title("Overlay"); axs[2].axis("off")
        plt.tight_layout(); plt.show()

show_samples(train_recs, n=3, seed=42, title="train")


# In[9]:


# ==== Dataset & DataLoaders ====
import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

IMG_SIZE   = 512
BATCH_SIZE = 2

class HRFDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = np.array(Image.open(rec["img"]).convert("RGB"))
        gt  = np.array(Image.open(rec["gt"]).convert("L"))
        gt  = (gt > 0).astype(np.uint8)
        fov = None
        if rec.get("fov"):
            fov = np.array(Image.open(rec["fov"]).convert("L"))
            fov = (fov > 0).astype(np.uint8)

        if self.transform is not None:
            aug = self.transform(image=img, mask=gt)
            img, gt = aug["image"], aug["mask"]
            if fov is not None:
                fov = A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_NEAREST)(image=fov)["image"]

        img = img.astype(np.float32) / 255.0
        gt  = gt.astype(np.float32)
        if fov is not None: fov = fov.astype(np.float32)

        out = {
            "image": torch.from_numpy(img.transpose(2,0,1)),
            "mask":  torch.from_numpy(gt).unsqueeze(0),
            "id":    rec["img"].stem
        }
        if fov is not None:
            out["fov"] = torch.from_numpy(fov).unsqueeze(0).float()
        return out

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(0.1,0.1,p=0.5),
])
val_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_LINEAR)])

pin = torch.cuda.is_available()
train_loader = DataLoader(HRFDataset(train_recs, transform=train_tf), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
val_loader   = DataLoader(HRFDataset(val_recs,   transform=val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)
test_loader  = DataLoader(HRFDataset(test_recs,  transform=val_tf),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

b = next(iter(train_loader))
print("Batch shapes:", tuple(b["image"].shape), tuple(b["mask"].shape), "FOV" if "fov" in b else "(no FOV)")


# In[10]:


# ========= Model, Loss, Metrics + 5-Fold Cross-Validation (HRF) =========
import os, time, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

# ----------------- fixed hyper-params (same as before) -----------------
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS        = 60           # <== epochs visible here
LEARNING_RATE = 1e-4
ACC_THR       = 0.5
OUTPUT_DIR    = "C:/Users/SC/Documents/Data and Code/HRF"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- model -----------------
def build_model():
    """UNet++ with ResNet34 encoder (ImageNet), 3→1, same as your previous setup."""
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(DEVICE)

# ----------------- losses & metrics -----------------
bce_loss = nn.BCEWithLogitsLoss().to(DEVICE)

def _apply_fov(t, fov):
    # t, fov: [B,1,H,W]
    return t * fov if fov is not None else t

def dice_coef(pred, target, fov=None, eps=1e-7):
    prob = torch.sigmoid(pred)
    prob   = _apply_fov(prob,   fov)
    target = _apply_fov(target, fov)
    num = 2.0 * (prob * target).sum(dim=(2,3))
    den = (prob + target).sum(dim=(2,3)) + eps
    return (num / den).mean()

def accuracy(pred, target, fov=None, thr=ACC_THR):
    prob  = torch.sigmoid(pred)
    predb = (prob > thr).float()
    predb  = _apply_fov(predb,  fov)
    target = _apply_fov(target, fov)
    return (predb == target).float().mean()

def criterion(pred, target, fov=None, bce_w=0.5):
    bce = bce_loss(_apply_fov(pred, fov), _apply_fov(target, fov))
    dsc = 1.0 - dice_coef(pred, target, fov=fov)
    return bce_w*bce + (1.0 - bce_w)*dsc

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tloss, tdice, tacc = [], [], []
    for batch in loader:
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)
        fov = batch.get("fov"); fov = fov.to(DEVICE) if fov is not None else None
        out = model(img)
        loss = criterion(out, msk, fov=fov)
        tloss.append(loss.item())
        tdice.append(dice_coef(out, msk, fov=fov).item())
        tacc.append(accuracy(out, msk, fov=fov).item())
    return {"loss": float(np.mean(tloss)),
            "dice": float(np.mean(tdice)),
            "acc":  float(np.mean(tacc))}

def train_one_epoch(model, loader, optimizer):
    model.train()
    tloss, tdice, tacc = [], [], []
    for batch in loader:
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)
        fov = batch.get("fov"); fov = fov.to(DEVICE) if fov is not None else None
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, msk, fov=fov)
        loss.backward()
        optimizer.step()
        tloss.append(loss.item())
        tdice.append(dice_coef(out, msk, fov=fov).item())
        tacc.append(accuracy(out, msk, fov=fov).item())
    return float(np.mean(tloss)), float(np.mean(tdice)), float(np.mean(tacc))

# ----------------- 5-fold cross validation -----------------
FOLDS = 5
SEED  = 42
pin   = torch.cuda.is_available()

# Use all available records for CV. We accept either:
#   - `pairs`  (single combined list of dicts: {"img","gt","fov"})
#   - or train/val/test lists you created before: `train_recs`, `val_recs`, `test_recs`
if 'pairs' in globals():
    pairs_all = list(pairs)
elif all(k in globals() for k in ['train_recs','val_recs','test_recs']):
    pairs_all = list(train_recs) + list(val_recs) + list(test_recs)
else:
    raise RuntimeError("Please define `pairs` OR (`train_recs`,`val_recs`,`test_recs`) before running CV.")

# HRFDataset, train_tf, val_tf, BATCH_SIZE must already be defined earlier.
def make_loaders(idx_train, idx_val):
    recs_tr = [pairs_all[i] for i in idx_train]
    recs_va = [pairs_all[i] for i in idx_val]
    dl_tr = DataLoader(HRFDataset(recs_tr, transform=train_tf), batch_size=BATCH_SIZE,
                       shuffle=True,  num_workers=0, pin_memory=pin)
    dl_va = DataLoader(HRFDataset(recs_va, transform=val_tf),   batch_size=BATCH_SIZE,
                       shuffle=False, num_workers=0, pin_memory=pin)
    return dl_tr, dl_va

# build fold bins
N = len(pairs_all)
indices = list(range(N))
random.Random(SEED).shuffle(indices)
fold_bins = np.array_split(indices, FOLDS)

cv_summary = {"per_fold": [], "mean": {}}

for f in range(FOLDS):
    print(f"\n==================== Fold {f+1}/{FOLDS} ====================")
    torch.manual_seed(SEED+f); np.random.seed(SEED+f); random.seed(SEED+f)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED+f)

    val_idx   = list(fold_bins[f])
    train_idx = [i for k in range(FOLDS) if k != f for i in fold_bins[k]]
    train_loader, val_loader = make_loaders(train_idx, val_idx)

    model     = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_acc":  [], "val_acc":  [],
        "lr": []
    }

    best_val  = float("inf")
    best_path = os.path.join(OUTPUT_DIR, f"unetpp_hrf_best_fold{f+1}.pth")

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        tr_loss, tr_dice, tr_acc = train_one_epoch(model, train_loader, optimizer)
        va_metrics = evaluate(model, val_loader)
        scheduler.step(va_metrics["loss"])

        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_metrics["loss"])
        history["val_dice"].append(va_metrics["dice"])
        history["val_acc"].append(va_metrics["acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if va_metrics["loss"] < best_val:
            best_val = va_metrics["loss"]
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, best_path)

        dt = time.time() - t0
        print(f"Fold {f+1} | Epoch {epoch:02d}/{EPOCHS} | "
              f"Train L {tr_loss:.4f} D {tr_dice:.3f} A {tr_acc:.3f} | "
              f"Val L {va_metrics['loss']:.4f} D {va_metrics['dice']:.3f} A {va_metrics['acc']:.3f} | "
              f"time {dt:.1f}s")

    # save per-fold history and evaluate best on its val set
    hist_csv = os.path.join(OUTPUT_DIR, f"history_fold{f+1}.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)
    print(f"✓ Saved {hist_csv}")
    print(f"✓ Best checkpoint -> {best_path}")

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    best_metrics = evaluate(model, val_loader)

    cv_summary["per_fold"].append({
        "fold": f+1,
        "loss": float(best_metrics["loss"]),
        "dice": float(best_metrics["dice"]),
        "acc":  float(best_metrics["acc"]),
        "best_ckpt": os.path.basename(best_path)
    })

    # free memory between folds
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# aggregate mean metrics across folds
cv_summary["mean"] = {
    "loss": float(np.mean([r["loss"] for r in cv_summary["per_fold"]])),
    "dice": float(np.mean([r["dice"] for r in cv_summary["per_fold"]])),
    "acc":  float(np.mean([r["acc"]  for r in cv_summary["per_fold"]]))
}

with open(os.path.join(OUTPUT_DIR, "cv5_val_summary.json"), "w") as f:
    json.dump(cv_summary, f, indent=2)

print("\n===== CV 5-Fold Validation Summary =====")
print(json.dumps(cv_summary, indent=2))
print("✓ Saved cv5_val_summary.json")


# In[11]:


# ========= Visualize 5-fold training history (HRF) =========
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- config (uses same OUTPUT_DIR and FOLDS you defined above) ---
assert os.path.isdir(OUTPUT_DIR), f"OUTPUT_DIR not found: {OUTPUT_DIR}"

# ---------- helpers ----------
def _load_fold_history(fold_idx, output_dir=OUTPUT_DIR):
    path = os.path.join(output_dir, f"history_fold{fold_idx}.csv")
    if not os.path.exists(path):
        print(f"[warn] missing: {path}")
        return None
    df = pd.read_csv(path)
    # add epoch column 1..N (in case it isn't there)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", np.arange(1, len(df)+1))
    return df

def _best_epoch(df, by="val_loss", mode="min"):
    if df is None or by not in df.columns: 
        return None, None
    if mode == "min":
        i = int(df[by].idxmin())
    else:
        i = int(df[by].idxmax())
    row = df.iloc[i].to_dict()
    return i+1, row  # epoch index is 1-based

def _overlay_plot(histories, key_train, key_val, title, ylabel):
    plt.figure(figsize=(7.5,5))
    for k, df in histories.items():
        if df is None: continue
        plt.plot(df["epoch"], df[key_train], alpha=0.65, lw=1.8, label=f"Fold {k} Train")
        plt.plot(df["epoch"], df[key_val],   alpha=0.85, lw=2.2, linestyle="--", label=f"Fold {k} Val")
    plt.title(title)
    plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=9)
    plt.tight_layout(); plt.show()

def _mean_std_plot(histories, key_train, key_val, title, ylabel):
    # stack to [folds, epochs]
    mats_tr, mats_va = [], []
    x = None
    for k, df in histories.items():
        if df is None: continue
        mats_tr.append(df[key_train].values)
        mats_va.append(df[key_val].values)
        x = df["epoch"].values
    if not mats_tr: 
        print("[warn] no histories found for mean±std plot")
        return
    tr = np.vstack(mats_tr)
    va = np.vstack(mats_va)
    plt.figure(figsize=(7.5,5))
    # train
    plt.plot(x, tr.mean(0), lw=2.5, label="Train (mean)")
    plt.fill_between(x, tr.mean(0)-tr.std(0), tr.mean(0)+tr.std(0), alpha=0.2, label="Train (±1σ)")
    # val
    plt.plot(x, va.mean(0), lw=2.5, linestyle="--", label="Val (mean)")
    plt.fill_between(x, va.mean(0)-va.std(0), va.mean(0)+va.std(0), alpha=0.2, label="Val (±1σ)")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()

def _plot_lr(histories):
    plt.figure(figsize=(7,4))
    for k, df in histories.items():
        if df is None or "lr" not in df.columns: continue
        plt.plot(df["epoch"], df["lr"], label=f"Fold {k}")
    plt.title("Learning Rate per Fold"); plt.xlabel("Epoch"); plt.ylabel("LR")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# ---------- load all fold histories ----------
histories = {f: _load_fold_history(f) for f in range(1, FOLDS+1)}

# ---------- quick per-fold best-epoch table ----------
rows = []
for f, df in histories.items():
    if df is None: 
        continue
    e_min, row_min = _best_epoch(df, by="val_loss", mode="min")
    e_max, row_max = _best_epoch(df, by="val_dice", mode="max")
    rows.append({
        "fold": f,
        "best_epoch_by_val_loss": e_min,
        "val_loss(best)": None if row_min is None else round(row_min["val_loss"], 4),
        "val_dice(at_best_loss)": None if row_min is None else round(row_min["val_dice"], 4),
        "best_epoch_by_val_dice": e_max,
        "val_dice(best)": None if row_max is None else round(row_max["val_dice"], 4),
        "val_acc(at_best_dice)": None if row_max is None else round(row_max["val_acc"], 4),
    })
best_table = pd.DataFrame(rows).sort_values("fold")
print("Per-fold best epochs:")
display(best_table)

# ---------- also show cv5_val_summary.json if present ----------
cv_sum_path = os.path.join(OUTPUT_DIR, "cv5_val_summary.json")
if os.path.exists(cv_sum_path):
    with open(cv_sum_path, "r") as f:
        cvsum = json.load(f)
    print("\nCV summary (from cv5_val_summary.json):")
    display(pd.DataFrame(cvsum["per_fold"]).sort_values("fold"))
    print("Mean across folds:", cvsum.get("mean", {}))
else:
    print("\n[warn] cv5_val_summary.json not found (it’s written at the end of CV training).")

# ---------- per-fold individual plots ----------
for f, df in histories.items():
    if df is None: 
        continue
    fig, axs = plt.subplots(1, 3, figsize=(14,4))
    axs[0].plot(df["epoch"], df["train_loss"], label="Train"); 
    axs[0].plot(df["epoch"], df["val_loss"],   label="Val", linestyle="--")
    axs[0].set_title(f"Fold {f} — Loss"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("BCE+Dice"); axs[0].grid(True); axs[0].legend()

    axs[1].plot(df["epoch"], df["train_dice"], label="Train"); 
    axs[1].plot(df["epoch"], df["val_dice"],   label="Val", linestyle="--")
    axs[1].set_title(f"Fold {f} — Dice"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Dice"); axs[1].grid(True); axs[1].legend()

    axs[2].plot(df["epoch"], df["train_acc"], label="Train"); 
    axs[2].plot(df["epoch"], df["val_acc"],   label="Val", linestyle="--")
    axs[2].set_title(f"Fold {f} — Accuracy"); axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("Accuracy"); axs[2].grid(True); axs[2].legend()

    plt.tight_layout(); plt.show()

# ---------- overlays: all folds on the same chart ----------
_overlay_plot(histories, "train_loss", "val_loss", "Loss — all folds", "BCE + Dice")
_overlay_plot(histories, "train_dice", "val_dice", "Dice — all folds", "Dice")
_overlay_plot(histories, "train_acc",  "val_acc",  "Accuracy — all folds", "Accuracy")

# ---------- mean ± std across folds ----------
_mean_std_plot(histories, "train_loss", "val_loss", "Loss — mean ± std across folds", "BCE + Dice")
_mean_std_plot(histories, "train_dice", "val_dice", "Dice — mean ± std across folds", "Dice")
_mean_std_plot(histories, "train_acc",  "val_acc",  "Accuracy — mean ± std across folds", "Accuracy")

# ---------- LR schedule ----------
_plot_lr(histories)


# In[12]:


# ========= CV-aware TEST evaluation + PREDICTIONS (HRF) =========
import os, json, glob, numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- ensure we have a test_loader (70/15/15 split made earlier) ---
pin = torch.cuda.is_available()
if 'test_loader' not in globals():
    assert 'test_recs' in globals(), "Define test_recs (list of records) first."
    test_loader = DataLoader(HRFDataset(test_recs, transform=val_tf),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

# --- helpers to load best/all fold models ---
def load_best_fold_model(select_metric="dice"):
    """
    Choose the best fold using cv5_val_summary.json (created during CV training).
    select_metric: 'dice' (max), 'acc' (max), or 'loss' (min).
    Returns: (model, best_fold_idx, best_row_dict)
    """
    summary_path = os.path.join(OUTPUT_DIR, "cv5_val_summary.json")
    assert os.path.exists(summary_path), f"Missing {summary_path} — run CV training first."
    with open(summary_path, "r") as f:
        summary = json.load(f)
    rows = summary["per_fold"]
    if select_metric.lower() == "loss":
        best = min(rows, key=lambda r: r["loss"])
    else:  # 'dice' or 'acc'
        best = max(rows, key=lambda r: r[select_metric.lower()])
    best_fold = int(best["fold"])
    ckpt_path = os.path.join(OUTPUT_DIR, f"unetpp_hrf_best_fold{best_fold}.pth")
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
    m = build_model()
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["state_dict"])
    m.to(DEVICE).eval()
    print(f"[BEST] Using fold {best_fold} by {select_metric}:",
          {k: round(v,4) for k,v in best.items() if k not in ["fold","best_ckpt"]})
    return m, best_fold, best

def load_all_fold_models():
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "unetpp_hrf_best_fold*.pth")))
    assert len(ckpts) >= 1, "No fold checkpoints found — run CV training first."
    models = []
    for ck in ckpts:
        m = build_model()
        m.load_state_dict(torch.load(ck, map_location=DEVICE)["state_dict"])
        m.to(DEVICE).eval()
        models.append(m)
    print(f"[ENSEMBLE] Loaded {len(models)} fold models.")
    return models

# --- (optional) evaluation with ensemble (average logits) ---
@torch.no_grad()
def evaluate_ensemble(models, loader):
    tloss, tdice, tacc = [], [], []
    for batch in loader:
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)
        fov = batch.get("fov")
        fov = fov.to(DEVICE) if fov is not None else None

        logits_sum = None
        for m in models:
            out = m(img)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        out = logits_sum / len(models)

        loss = criterion(out, msk, fov=fov)
        tloss.append(loss.item())
        tdice.append(dice_coef(out, msk, fov=fov).item())
        tacc.append(accuracy(out, msk, fov=fov).item())

    return {"loss": float(np.mean(tloss)),
            "dice": float(np.mean(tdice)),
            "acc":  float(np.mean(tacc))}

# --- utilities to save predictions at ORIGINAL size ---
def _resize_to_original(arr, target_w, target_h, is_mask):
    if arr.ndim == 2:
        inter = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(arr, (target_w, target_h), interpolation=inter)
    elif arr.ndim == 3:
        inter = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(arr, (target_w, target_h), interpolation=inter)
    else:
        raise ValueError("Unsupported array shape for resize.")

def _get_orig_hw(img_id):
    # use true original image to get H,W
    assert 'ID2PATH_TEST' in globals(), "ID2PATH_TEST not defined — build it from test_recs earlier."
    path = ID2PATH_TEST.get(img_id, None)
    if path is None or not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im.close()
    return (h, w)

@torch.no_grad()
def save_test_predictions_bestfold(out_dir, thr=0.5, save_probs=False, apply_fov=True, resize_to_original=True):
    os.makedirs(out_dir, exist_ok=True)
    model, best_fold, _ = load_best_fold_model(select_metric="dice")
    print(f"[SAVE] Best-fold predictions -> {out_dir}")
    for batch in test_loader:
        imgs = batch["image"].to(DEVICE)
        ids  = batch["id"]
        fov  = batch.get("fov")
        fov  = fov.to(DEVICE) if fov is not None else None

        logits = model(imgs)
        probs  = torch.sigmoid(logits)
        if apply_fov and (fov is not None):
            probs = probs * fov

        probs_np = probs.squeeze(1).cpu().numpy()   # [B,H,W]
        preds_np = (probs_np > thr).astype(np.uint8) * 255

        for i, img_id in enumerate(ids):
            # resize back to HRF original size
            out_prob = probs_np[i]
            out_mask = preds_np[i]
            if resize_to_original:
                hw = _get_orig_hw(img_id)
                if hw is not None:
                    H, W = hw
                    out_prob = _resize_to_original(out_prob, W, H, is_mask=False)
                    out_mask = _resize_to_original(out_mask, W, H, is_mask=True)

            # save
            mask_path = os.path.join(out_dir, f"{img_id}_pred_best.png")
            cv2.imwrite(mask_path, out_mask)
            if save_probs:
                prob_path = os.path.join(out_dir, f"{img_id}_prob_best.png")
                cv2.imwrite(prob_path, (np.clip(out_prob,0,1)*255).astype(np.uint8))

@torch.no_grad()
def save_test_predictions_ensemble(out_dir, thr=0.5, save_probs=False, apply_fov=True, resize_to_original=True):
    os.makedirs(out_dir, exist_ok=True)
    models = load_all_fold_models()
    print(f"[SAVE] Ensemble predictions -> {out_dir}")
    for batch in test_loader:
        imgs = batch["image"].to(DEVICE)
        ids  = batch["id"]
        fov  = batch.get("fov")
        fov  = fov.to(DEVICE) if fov is not None else None

        logits_sum = None
        for m in models:
            out = m(imgs)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        logits = logits_sum / len(models)
        probs  = torch.sigmoid(logits)
        if apply_fov and (fov is not None):
            probs = probs * fov

        probs_np = probs.squeeze(1).cpu().numpy()
        preds_np = (probs_np > thr).astype(np.uint8) * 255

        for i, img_id in enumerate(ids):
            out_prob = probs_np[i]
            out_mask = preds_np[i]
            if resize_to_original:
                hw = _get_orig_hw(img_id)
                if hw is not None:
                    H, W = hw
                    out_prob = _resize_to_original(out_prob, W, H, is_mask=False)
                    out_mask = _resize_to_original(out_mask, W, H, is_mask=True)

            mask_path = os.path.join(out_dir, f"{img_id}_pred_ens.png")
            cv2.imwrite(mask_path, out_mask)
            if save_probs:
                prob_path = os.path.join(out_dir, f"{img_id}_prob_ens.png")
                cv2.imwrite(prob_path, (np.clip(out_prob,0,1)*255).astype(np.uint8))

# --- TEST evaluation (best fold and ensemble) ---
# Evaluate best fold on TEST
best_model, best_fold, _ = load_best_fold_model(select_metric="dice")
test_metrics_best = evaluate(best_model, test_loader)
print("\n[TEST] Best fold metrics:")
for k, v in test_metrics_best.items():
    print(f"  {k}: {v:.4f}")
with open(os.path.join(OUTPUT_DIR, f"test_metrics_best_fold{best_fold}.json"), "w") as f:
    json.dump({k: float(v) for k, v in test_metrics_best.items()}, f, indent=2)

# Evaluate ensemble on TEST (optional, often slightly better)
models_ens = load_all_fold_models()
test_metrics_ens = evaluate_ensemble(models_ens, test_loader)
print("\n[TEST] Ensemble metrics:")
for k, v in test_metrics_ens.items():
    print(f"  {k}: {v:.4f}")
with open(os.path.join(OUTPUT_DIR, "test_metrics_ensemble.json"), "w") as f:
    json.dump({k: float(v) for k, v in test_metrics_ens.items()}, f, indent=2)

# --- Save predictions to disk ---
save_test_predictions_bestfold(
    out_dir=os.path.join(OUTPUT_DIR, "preds_best"),
    thr=0.5, save_probs=False, apply_fov=True, resize_to_original=True
)
save_test_predictions_ensemble(
    out_dir=os.path.join(OUTPUT_DIR, "preds_ens"),
    thr=0.5, save_probs=False, apply_fov=True, resize_to_original=True
)

# --- Quick visualization of a few test samples (best fold) ---
@torch.no_grad()
def _prepare_for_show(x_np):
    x = x_np.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    return np.clip(x, 0.0, 1.0)

@torch.no_grad()
def visualize_test_samples_best(n=6, thr=0.5):
    """
    Visualization ONLY (no FoV masking here): Original | Pred | GT
    """
    model, best_fold, _ = load_best_fold_model(select_metric="dice")
    shown = 0
    for batch in test_loader:
        imgs = batch["image"].to(DEVICE)
        ids  = batch["id"]
        gt   = batch["mask"].cpu().numpy()

        logits = model(imgs)
        probs  = torch.sigmoid(logits)            # <-- NO FoV masking here
        preds  = (probs > thr).float().cpu().numpy()

        imgs_np = imgs.detach().cpu().numpy().transpose(0, 2, 3, 1)
        for i in range(imgs.size(0)):
            if shown >= n:
                return
            # show true original if we can
            orig = None
            if 'ID2PATH_TEST' in globals():
                p = ID2PATH_TEST.get(ids[i], None)
                if p and os.path.exists(p):
                    orig = np.array(Image.open(p).convert("RGB"))
            if orig is None:
                orig = (imgs_np[i] * 255.0).astype(np.uint8)

            fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
            axs[0].imshow(_prepare_for_show(orig)); axs[0].set_title(f"Original: {ids[i]}"); axs[0].axis("off")
            axs[1].imshow(preds[i, 0], cmap="gray"); axs[1].set_title("Pred");               axs[1].axis("off")
            axs[2].imshow(gt[i, 0],   cmap="gray");   axs[2].set_title("GT");                 axs[2].axis("off")
            plt.tight_layout(); plt.show()
            shown += 1

# Example: show 6 predictions from TEST using the best fold (no FoV masking in visualization)
visualize_test_samples_best(n=6, thr=0.5)


# In[ ]:





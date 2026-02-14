#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =========================================
# Cell 1 — Config (EDIT THESE 2 PATHS)
# =========================================
from pathlib import Path

# CHASEDB1 paths
IMG_DIR  = Path(r"C:\Users\SC\Documents\Data and Code\chasedb1 dataset\Images")  # e.g., Image_01L.jpg / .png / .tif
MASK_DIR = Path(r"C:\Users\SC\Documents\Data and Code\chasedb1 dataset\Masks")   # e.g., Image_01L_1stHO.png (binary or colored)

# (Optional) If you have a dedicated test set, set these; otherwise leave = None
TEST_IMG_DIR  = None
TEST_MASK_DIR = None

OUT_ROOT = Path(r"C:\Users\SC\Documents\Data and Code\chasedb1 dataset")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Training hyperparams (kept same as your working notebook)
SEED          = 42
FOLDS         = 5                 # CV folds
IMG_SIZE      = (512, 512)        # (H, W)
BATCH_SIZE    = 2
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
MAX_EPOCHS    = 100
THRESHOLD     = 0.5
NUM_WORKERS   = 0                 # Windows-safe
PIN_MEMORY    = False

DEVICE_FALLBACK = "cuda"          # use CUDA if available, else CPU
MODEL_NAME = "nvidia/segformer-b1-finetuned-ade-512-512"  # good starting backbone
CKPT_NAME  = "best_segformer.pt"


# In[2]:


# =========================================
# Cell 2 — Imports & Reproducibility
# =========================================
import os, time, random, warnings, math
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import SegformerForSemanticSegmentation

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

DEVICE = torch.device("cuda" if torch.cuda.is_available() and DEVICE_FALLBACK=="cuda" else "cpu")
PIN_MEMORY = PIN_MEMORY and (DEVICE.type=="cuda")
print("Device:", DEVICE)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# In[3]:


# =========================================
# Cell 3 — Robust readers & pairing helpers
# =========================================
IMG_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".ppm",".gif"}

def is_img(p: Path): return p.is_file() and p.suffix.lower() in IMG_EXTS

def pil_read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA",):
            im = im.convert("RGB")
        elif im.mode in ("L",):
            im = im.convert("RGB")
        elif im.mode in ("I;16","I"):
            arr = np.array(im, dtype=np.float32)
            mx = float(arr.max()) if arr.max()>0 else 1.0
            arr = (arr/mx*255).clip(0,255).astype(np.uint8)
            im  = Image.fromarray(arr).convert("RGB")
        elif im.mode != "RGB":
            im = im.convert("RGB")
        return np.array(im)

def pil_read_mask(path: Path) -> np.ndarray:
    """
    Return binary mask (0/1). Works if mask is grayscale or colored (non-black = vessel).
    """
    with Image.open(path) as im:
        im.load()
        arr = np.array(im)
    if arr.ndim==3:
        mask = (arr.max(axis=-1)>0).astype(np.uint8)
    else:
        mask = (arr>0).astype(np.uint8)
    return mask

# Common mask suffixes to strip for matching (CHASEDB1 often has *_1stHO, *_vessel, etc.)
MASK_SUFFIXES = ["_mask", "_1stHO", "_manual1", "_vessel", "-vessels", "_binary"]

def _strip_suffixes(stem: str) -> str:
    for suf in MASK_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem

def pair_images_and_masks(img_dir: Path, mask_dir: Path):
    imgs = sorted([p for p in img_dir.rglob("*") if is_img(p)])
    masks = sorted([p for p in mask_dir.rglob("*") if is_img(p)])
    mask_map = {}
    for mp in masks:
        mask_map.setdefault(_strip_suffixes(mp.stem), []).append(mp)

    paired_imgs, paired_masks = [], []
    misses = 0
    for ip in imgs:
        key = _strip_suffixes(ip.stem)
        cand = mask_map.get(key, None)
        if not cand:
            # loose: try exact stem, then any with same prefix
            cand = [mp for mp in masks if mp.stem==ip.stem] or \
                   [mp for mp in masks if _strip_suffixes(mp.stem)==key]
        if cand:
            paired_imgs.append(ip); paired_masks.append(cand[0])
        else:
            misses += 1
    if misses:
        print(f"[WARN] {misses} images had no mask in {mask_dir} (skipped).")
    return paired_imgs, paired_masks


# In[4]:


# =========================================
# Cell 4 — Albumentations & Dataset
# =========================================
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

class FundusDataset(Dataset):
    def __init__(self, img_paths, mask_paths=None, train=False):
        self.imgs  = list(img_paths)
        self.masks = list(mask_paths) if mask_paths is not None else None
        self.tf    = train_tf if train else val_tf
        self.train = train

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        ip = Path(self.imgs[i])
        img = pil_read_rgb(ip)

        if self.masks is not None:
            gp = Path(self.masks[i])
            m  = pil_read_mask(gp)        # 0/1 numpy
            aug = self.tf(image=img, mask=m)
            x   = aug["image"].float()    # [3,H,W]
            y   = aug["mask"].unsqueeze(0).float()  # [1,H,W]
            return x, y, str(ip), str(gp)
        else:
            aug = self.tf(image=img)
            x   = aug["image"].float()
            return x, None, str(ip), None


# In[5]:


# =========================================
# Cell 5 — Metrics & Loss
# =========================================
def binarize(logits, thr=THRESHOLD):
    return (torch.sigmoid(logits) > thr).float()

def confusion_components(pred, target):
    tp = (pred*target).sum().item()
    tn = ((1-pred)*(1-target)).sum().item()
    fp = (pred*(1-target)).sum().item()
    fn = ((1-pred)*target).sum().item()
    return tp, tn, fp, fn

def metrics_from_conf(tp, tn, fp, fn, eps=1e-8):
    acc  = (tp+tn)/(tp+tn+fp+fn+eps)
    prec = tp/(tp+fp+eps)
    sens = tp/(tp+fn+eps)
    spec = tn/(tn+fp+eps)
    f1   = 2*prec*sens/(prec+sens+eps)
    dice = f1
    return dict(acc=acc, precision=prec, sensitivity=sens, specificity=spec, f1=f1, dice=dice)

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bw  = bce_w
        self.dw  = dice_w
    def forward(self, logits, targets, eps=1e-6):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2.0*(probs*targets).sum(dim=(1,2,3))
        den = (probs+targets).sum(dim=(1,2,3)) + eps
        dice = 1.0 - (num/den).mean()
        return self.bw*bce + self.dw*dice

loss_fn = BCEDiceLoss()


# In[6]:


# =========================================
# Cell 6 — Model builder
# =========================================
def build_model():
    # 2 labels (bg, vessel); we’ll use channel 1 as vessel logits
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    return model.to(DEVICE)


# In[7]:


# =========================================
# Cell 7 — Load & split CHASEDB1 (70/15/15) + KFold on train(70%)
# =========================================
all_imgs, all_masks = pair_images_and_masks(IMG_DIR, MASK_DIR)
print(f"Found paired images: {len(all_imgs)}")

if TEST_IMG_DIR is None:
    # Split 70/15/15
    idx = np.arange(len(all_imgs))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=SEED, shuffle=True)
    va_idx, te_idx  = train_test_split(tmp_idx, test_size=0.50, random_state=SEED, shuffle=True)
    split = {
        "train": ([all_imgs[i] for i in tr_idx], [all_masks[i] for i in tr_idx]),
        "val":   ([all_imgs[i] for i in va_idx], [all_masks[i] for i in va_idx]),
        "test":  ([all_imgs[i] for i in te_idx], [all_masks[i] for i in te_idx]),
    }
else:
    # Use external test set if provided
    te_imgs, te_masks = pair_images_and_masks(TEST_IMG_DIR, TEST_MASK_DIR) if TEST_MASK_DIR else (sorted([p for p in TEST_IMG_DIR.rglob("*") if is_img(p)]), [None]*999)
    split = {
        "train": (all_imgs, all_masks),
        "val":   ([], []),      # will be created per fold
        "test":  (te_imgs, te_masks),
    }

train_imgs, train_gts = split["train"]
val_imgs_base, val_gts_base = split["val"]
test_imgs, test_gts = split["test"]
print(f"Split sizes — train:{len(train_imgs)}  val:{len(val_imgs_base)}  test:{len(test_imgs)}")


# In[8]:


# =========================================
# Cell 8 — KFold CV on the 70% train split
# Saves best checkpoint per fold + history.csv
# =========================================
from torch.optim import AdamW

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
hist_cols = ["epoch","train_loss","train_dice","train_acc","train_prec","train_spec","train_sens","train_f1",
             "val_loss","val_dice","val_acc","val_prec","val_spec","val_sens","val_f1"]

fold_summaries = []

train_imgs_np = np.array(train_imgs)
train_gts_np  = np.array(train_gts)

for fold, (tr_idx, va_idx) in enumerate(kf.split(train_imgs_np), start=1):
    print(f"\n===== FOLD {fold}/{FOLDS} =====")
    fold_dir = OUT_ROOT / f"fold_{fold:02d}"
    (fold_dir/"checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir/"logs").mkdir(parents=True, exist_ok=True)
    (fold_dir/"pred_val").mkdir(parents=True, exist_ok=True)

    tr_imgs = train_imgs_np[tr_idx].tolist()
    tr_gts  = train_gts_np[tr_idx].tolist()
    va_imgs = train_imgs_np[va_idx].tolist()
    va_gts  = train_gts_np[va_idx].tolist()

    ds_tr = FundusDataset(tr_imgs, tr_gts, train=True)
    ds_va = FundusDataset(va_imgs, va_gts, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

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
            out = model(pixel_values=xb).logits
            if out.shape[-2:] != yb.shape[-2:]:
                out = F.interpolate(out, size=yb.shape[-2:], mode="bilinear", align_corners=False)
            vessel_logits = out[:,1:2,:,:]

            loss = loss_fn(vessel_logits, yb)
            loss.backward()
            optimizer.step()

            run_loss += float(loss.item()) * xb.size(0)
            with torch.no_grad():
                pred = binarize(vessel_logits, THRESHOLD)
                tp, tn, fp, fn = confusion_components(pred, yb)
                mets = metrics_from_conf(tp, tn, fp, fn)
                for k in t_metrics: t_metrics[k] += mets[k]*xb.size(0)
                n_seen += xb.size(0)

        train_loss = run_loss / max(1,n_seen)
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
                    out = F.interpolate(out, size=yb.shape[-2:], mode="bilinear", align_corners=False)
                vessel_logits = out[:,1:2,:,:]
                loss = loss_fn(vessel_logits, yb)
                v_loss += float(loss.item())*xb.size(0)

                pred = binarize(vessel_logits, THRESHOLD)
                tp, tn, fp, fn = confusion_components(pred, yb)
                mets = metrics_from_conf(tp, tn, fp, fn)
                for k in v_metrics: v_metrics[k] += mets[k]*xb.size(0)
                n_seen_v += xb.size(0)

        val_loss = v_loss / max(1,n_seen_v)
        val_mets = {k: (v_metrics[k]/max(1,n_seen_v)) for k in v_metrics}

        # Log & save history
        row = [epoch,
               train_loss, train_mets["dice"], train_mets["acc"], train_mets["precision"], train_mets["specificity"], train_mets["sensitivity"], train_mets["f1"],
               val_loss,   val_mets["dice"],   val_mets["acc"],   val_mets["precision"],   val_mets["specificity"],   val_mets["sensitivity"],   val_mets["f1"]]
        history.append(row)
        pd.DataFrame(history, columns=hist_cols).to_csv(fold_dir/"logs"/"history.csv", index=False)

        # Save best ckpt by val dice
        if val_mets["dice"] > best_val_dice:
            best_val_dice = val_mets["dice"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "metric": float(best_val_dice),
                "config": {"IMG_SIZE": IMG_SIZE, "THRESHOLD": THRESHOLD}
            }, fold_dir/"checkpoints"/CKPT_NAME)
            print(f"[Fold {fold}] Epoch {epoch}: new best val Dice={best_val_dice:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            print(f"[Fold {fold}] Ep {epoch:03d} | tr_loss={train_loss:.4f} va_loss={val_loss:.4f} | tr_dice={train_mets['dice']:.4f} va_dice={val_mets['dice']:.4f}")

    fold_summaries.append({"fold": fold, "best_val_dice": best_val_dice})
    pd.DataFrame(fold_summaries).to_csv(OUT_ROOT/"cv_summary.csv", index=False)

print("\n[CV] Summary:")
print(pd.DataFrame(fold_summaries))


# In[8]:


# =========================================
# Cell 9 — Inference on TEST with best model of each fold
# Saves per-image Dice + inference time (ms) and a fold summary
# =========================================
import time

def save_pred_u8(t: torch.Tensor, path: Path):
    p = (t.detach().cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(p, mode="L").save(path)

class _TestSet(Dataset):
    def __init__(self, imgs, gts=None, tf=None):
        self.imgs = list(imgs)
        self.gts  = list(gts) if gts else None
        self.tf   = tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        ip  = Path(self.imgs[i])
        img = pil_read_rgb(ip)
        if self.gts:
            gp = Path(self.gts[i]); gt = pil_read_mask(gp)
            aug = val_tf(image=img, mask=gt)
            x = aug["image"].float()
            y = aug["mask"].unsqueeze(0).float()
            return x, y, str(ip), str(gp)
        else:
            aug = val_tf(image=img)
            x = aug["image"].float()
            return x, None, str(ip), None

def run_test_for_fold(fold_dir: Path, imgs, gts):
    ckpt_path = fold_dir / "checkpoints" / CKPT_NAME
    if not ckpt_path.exists():
        print(f"[SKIP] No checkpoint in {fold_dir}")
        return

    model = build_model()
    state = torch.load(ckpt_path, map_location=DEVICE)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds_test = _TestSet(imgs, gts, tf=val_tf)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)

    # (optional) warm-up for stable GPU timing
    if DEVICE.type == "cuda":
        with torch.no_grad():
            for xb, *_ in dl_test:
                _ = model(pixel_values=xb.to(DEVICE)).logits
                break
        torch.cuda.synchronize()

    pred_dir = fold_dir / "pred_test_best"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for xb, yb, ipath, gpath in dl_test:
            xb = xb.to(DEVICE, non_blocking=True)

            # ---- time forward pass ----
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(pixel_values=xb).logits
            logits = F.interpolate(logits, size=xb.shape[-2:], mode="bilinear", align_corners=False)
            vessel_logits = logits[:, 1:2, :, :]
            pred = binarize(vessel_logits, THRESHOLD)

            if DEVICE.type == "cuda": torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            out_path = pred_dir / f"{Path(ipath[0]).stem}_pred.png"
            save_pred_u8(pred[0, 0].cpu(), out_path)

            dice = np.nan
            if yb is not None:
                yb = yb.to(DEVICE, non_blocking=True)
                tp, tn, fp, fn = confusion_components(pred, yb)
                dice = metrics_from_conf(tp, tn, fp, fn)["dice"]

            rows.append({
                "image": ipath[0],
                "gt": (gpath[0] if gpath else None),
                "pred": str(out_path),
                "dice": float(dice) if not np.isnan(dice) else np.nan,
                "latency_ms": float(latency_ms)
            })

    # per-image CSV (now includes latency_ms)
    df = pd.DataFrame(rows)
    df.to_csv(fold_dir / "test_metrics_per_image.csv", index=False)
    print(f"[{fold_dir.name}] wrote:", fold_dir / "test_metrics_per_image.csv")

    # small per-fold summary
    df_summary = pd.DataFrame([{
        "fold": fold_dir.name,
        "mean_dice": df["dice"].dropna().mean(),
        "mean_latency_ms": df["latency_ms"].mean(),
        "n_images": len(df)
    }])
    df_summary.to_csv(fold_dir / "test_summary.csv", index=False)

# run across folds
for fd in sorted(OUT_ROOT.glob("fold_*")):
    run_test_for_fold(fd, test_imgs, test_gts)


# In[9]:


# =========================================
# Cell 9 — Inference on TEST with best model of each fold
# Saves per-image metrics + panels
# =========================================
def save_pred_u8(t: torch.Tensor, path: Path):
    p = (t.detach().cpu().numpy()*255).astype(np.uint8)
    Image.fromarray(p, mode="L").save(path)

class _TestSet(Dataset):
    def __init__(self, imgs, gts=None, tf=None):
        self.imgs=list(imgs); self.gts=list(gts) if gts else None; self.tf=tf
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        ip = Path(self.imgs[i]); img = pil_read_rgb(ip)
        if self.gts:
            gp = Path(self.gts[i]); gt = pil_read_mask(gp)
            aug = val_tf(image=img, mask=gt)
            x = aug["image"].float()
            y = aug["mask"].unsqueeze(0).float()
            return x, y, str(ip), str(gp)
        else:
            aug = val_tf(image=img)
            x = aug["image"].float()
            return x, None, str(ip), None

def run_test_for_fold(fold_dir: Path, imgs, gts):
    ckpt_path = fold_dir/"checkpoints"/CKPT_NAME
    if not ckpt_path.exists():
        print(f"[SKIP] No checkpoint in {fold_dir}")
        return

    model = build_model()
    state = torch.load(ckpt_path, map_location=DEVICE)
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds_test = _TestSet(imgs, gts, tf=val_tf)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)

    pred_dir = fold_dir/"pred_test_best"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows=[]
    with torch.no_grad():
        for xb, yb, ipath, gpath in dl_test:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(pixel_values=xb).logits
            logits = F.interpolate(logits, size=xb.shape[-2:], mode="bilinear", align_corners=False)
            vessel_logits = logits[:,1:2,:,:]
            pred = binarize(vessel_logits, THRESHOLD)

            out_path = pred_dir / f"{Path(ipath[0]).stem}_pred.png"
            save_pred_u8(pred[0,0].cpu(), out_path)

            dice = np.nan
            if yb is not None:
                yb = yb.to(DEVICE)
                tp, tn, fp, fn = confusion_components(pred, yb)
                dice = metrics_from_conf(tp, tn, fp, fn)["dice"]

            rows.append({"image": ipath[0], "gt": (gpath[0] if gpath else None),
                         "pred": str(out_path), "dice": float(dice) if not np.isnan(dice) else np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(fold_dir/"test_metrics_per_image.csv", index=False)
    print(f"[{fold_dir.name}] wrote:", fold_dir/"test_metrics_per_image.csv")

# run across folds
for fd in sorted(OUT_ROOT.glob("fold_*")):
    run_test_for_fold(fd, test_imgs, test_gts)


# In[10]:


# =========================================
# Cell 10 — Panels (Image | GT | Pred) & per-fold history plots
# =========================================
def resize_np(arr: np.ndarray, hw: tuple[int,int], is_mask=False):
    H,W = hw
    im = Image.fromarray(arr.astype(np.uint8))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return np.array(im.resize((W,H), resample=resample))

def make_fold_panel(fold_dir: Path, n_examples=5, mode="bestworst"):
    csv_path = fold_dir/"test_metrics_per_image.csv"
    if not csv_path.exists(): 
        print(f"[SKIP] No CSV in {fold_dir}"); return
    df = pd.read_csv(csv_path)
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")

    if mode=="bestworst" and df["dice"].notna().sum()>=n_examples:
        k1 = max(1, n_examples//2); k2 = n_examples-k1
        sel = pd.concat([df.nsmallest(k1,"dice"), df.nlargest(k2,"dice")], ignore_index=True)
    else:
        sel = df.sample(n=min(n_examples, len(df)), random_state=42)

    H,W = IMG_SIZE
    fig, axes = plt.subplots(len(sel), 3, figsize=(12, 3.2*len(sel)), dpi=130)
    if len(sel)==1: axes = np.expand_dims(axes, 0)

    for i,row in enumerate(sel.itertuples(index=False)):
        img  = resize_np(pil_read_rgb(Path(row.image)), (H,W), is_mask=False)
        pred = resize_np(np.array(Image.open(row.pred)), (H,W), is_mask=True)

        axes[i,0].imshow(img);  axes[i,0].set_title("Image"); axes[i,0].axis("off")
        if isinstance(row.gt,str) and len(row.gt):
            gt = resize_np(pil_read_mask(Path(row.gt))*255, (H,W), is_mask=True)
            axes[i,1].imshow(gt, cmap="gray"); axes[i,1].set_title("GT")
        else:
            axes[i,1].text(0.5,0.5,"No GT",ha="center",va="center"); axes[i,1].set_title("GT")
        axes[i,1].axis("off")

        t = "" if (isinstance(row.dice,float) and np.isnan(row.dice)) else f" (Dice={row.dice:.3f})"
        axes[i,2].imshow(pred, cmap="gray"); axes[i,2].set_title(f"Prediction{t}"); axes[i,2].axis("off")

    plt.tight_layout()
    panel_path = fold_dir / f"{fold_dir.name}_test_panel.png"
    fig.savefig(panel_path, bbox_inches="tight"); plt.show()
    print(f"[{fold_dir.name}] panel saved:", panel_path)

# Panels
for fd in sorted(OUT_ROOT.glob("fold_*")):
    make_fold_panel(fd, n_examples=5, mode="bestworst")

# Per-fold history
for fold in range(1, FOLDS+1):
    hist_path = OUT_ROOT / f"fold_{fold:02d}" / "logs" / "history.csv"
    if not hist_path.exists(): 
        continue
    hist = pd.read_csv(hist_path)
    plt.figure(figsize=(12,4), dpi=130)
    plt.subplot(1,2,1); plt.plot(hist["epoch"], hist["train_loss"], label="train"); plt.plot(hist["epoch"], hist["val_loss"], label="val")
    plt.title(f"Fold {fold} — Loss"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1,2,2); plt.plot(hist["epoch"], hist["train_dice"], label="train"); plt.plot(hist["epoch"], hist["val_dice"], label="val")
    plt.title(f"Fold {fold} — Dice"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)
    plt.show()


# In[11]:


# =========================================
# Cell 11 — Mean ± SD curves across folds (Loss & Dice)
# =========================================
def _collect_histories(root: Path):
    hs=[]
    for f in sorted(root.glob("fold_*")):
        p=f/"logs"/"history.csv"
        if p.exists():
            df=pd.read_csv(p)
            hs.append(df)
    return hs

hists=_collect_histories(OUT_ROOT)
if hists:
    # Align by epoch length (truncate to min length)
    minT=min(len(h) for h in hists)
    T = np.arange(1, minT+1)
    def stack(col): return np.stack([h[col].values[:minT] for h in hists], axis=0)

    tr_loss = stack("train_loss"); va_loss = stack("val_loss")
    tr_dice = stack("train_dice"); va_dice = stack("val_dice")

    plt.figure(figsize=(12,4), dpi=130)
    plt.subplot(1,2,1)
    plt.plot(T, tr_loss.mean(0), label="train μ")
    plt.fill_between(T, tr_loss.mean(0)-tr_loss.std(0), tr_loss.mean(0)+tr_loss.std(0), alpha=0.2, label="train ±σ")
    plt.plot(T, va_loss.mean(0), label="val μ")
    plt.fill_between(T, va_loss.mean(0)-va_loss.std(0), va_loss.mean(0)+va_loss.std(0), alpha=0.2, label="val ±σ")
    plt.title("Loss — mean ± SD"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(T, tr_dice.mean(0), label="train μ")
    plt.fill_between(T, tr_dice.mean(0)-tr_dice.std(0), tr_dice.mean(0)+tr_dice.std(0), alpha=0.2, label="train ±σ")
    plt.plot(T, va_dice.mean(0), label="val μ")
    plt.fill_between(T, va_dice.mean(0)-va_dice.std(0), va_dice.mean(0)+va_dice.std(0), alpha=0.2, label="val ±σ")
    plt.title("Dice — mean ± SD"); plt.xlabel("Epoch"); plt.legend(); plt.grid(alpha=0.3)
    plt.show()
else:
    print("[INFO] No histories found yet.")


# In[12]:


# =========================================
# Cell 12 — (Optional) Re-run inference with the single best fold (e.g., fold 03)
# Set BEST_FOLD to the index you found best in cv_summary.csv
# =========================================
BEST_FOLD = 4  # you said fold 3 performed best

best_dir = OUT_ROOT / f"fold_{BEST_FOLD:02d}"
run_test_for_fold(best_dir, test_imgs, test_gts)
make_fold_panel(best_dir, n_examples=6, mode="bestworst")


# In[ ]:





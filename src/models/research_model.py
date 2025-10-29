"""
Research 3D patch-based Transformer segmentation (robust + low-VRAM friendly)

Expect data layout:
  data/features/train/patient_xxx/image.pt  # shape like [160,192,155,4] (H,W,D,C)
  data/features/train/patient_xxx/mask.pt   # shape like [160,192,155,1]

Outputs:
  models/research/research_model.pt
  models/research/research_model_lite.pt (TorchScript)
Logs metrics & artifacts to MLflow (mlruns/)
"""

import os
import random
import time
import logging
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# ----------------------------
# Configuration (tweak if needed)
# ----------------------------
DATA_ROOT = "data/features"
TRAIN_SUBDIR = "train"
VAL_SUBDIR = "val"

MODEL_DIR = "models/research"
os.makedirs(MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "research_model.pt")
LITE_MODEL_PATH = os.path.join(MODEL_DIR, "research_model_lite.pt")

# Conservative defaults for low-VRAM GPUs (change if you have more VRAM)
PATCH_SIZE: Tuple[int,int,int] = (16, 64, 64)   # (depth, height, width)
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATCHES_PER_VOLUME = 4   # random patches per volume per epoch

NUM_WORKERS = 0  # use 0 on Windows to avoid DataLoader worker issues
USE_AMP = torch.cuda.is_available()  # enable AMP only if CUDA available

MLFLOW_EXPERIMENT = "NeuroLens_Research_Model"
MLFLOW_URI = "file:./mlruns"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Logging
# ----------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/research_model.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console)
logging.info(f"research_model.py starting. Device={DEVICE}, AMP={USE_AMP}")

# ----------------------------
# Utilities: Dice / IoU for logits (B,1,D,H,W)
# ----------------------------
def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    t = target.float()
    inter = (preds * t).sum(dim=(1,2,3,4))
    denom = preds.sum(dim=(1,2,3,4)) + t.sum(dim=(1,2,3,4))
    dice = ((2.0 * inter + eps) / (denom + eps)).mean().item()
    return dice

def iou_from_logits(logits: torch.Tensor, target: torch.Tensor, eps=1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    t = target.float()
    inter = (preds * t).sum(dim=(1,2,3,4))
    union = preds.sum(dim=(1,2,3,4)) + t.sum(dim=(1,2,3,4)) - inter
    iou = ((inter + eps) / (union + eps)).mean().item()
    return iou

# ----------------------------
# Dataset: PatchVolumeDataset
# ----------------------------
class VolumeIndex:
    def __init__(self, image_path: str, mask_path: str):
        self.image_path = image_path
        self.mask_path = mask_path

class PatchVolumeDataset(Dataset):
    """
    Samples random 3D patches from patient volumes.
    __len__ = num_volumes * patches_per_volume (so one epoch = this many patches)
    """
    def __init__(self, root_dir: str, patches_per_volume: int = PATCHES_PER_VOLUME, patch_size: Tuple[int,int,int] = PATCH_SIZE):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.patches_per_volume = max(1, int(patches_per_volume))
        self.volumes: List[VolumeIndex] = []
        if os.path.exists(self.root_dir):
            for name in sorted(os.listdir(self.root_dir)):
                pdir = os.path.join(self.root_dir, name)
                if not os.path.isdir(pdir):
                    continue
                ip = os.path.join(pdir, "image.pt")
                mp = os.path.join(pdir, "mask.pt")
                if os.path.exists(ip) and os.path.exists(mp):
                    self.volumes.append(VolumeIndex(ip, mp))
        else:
            logging.warning(f"PatchVolumeDataset root_dir not found: {self.root_dir}")

        # build index mapping each dataset index -> volume index
        self._build_index()
        logging.info(f"PatchVolumeDataset: found {len(self.volumes)} volumes under {self.root_dir}; index size {len(self.index_map)}")

    def _build_index(self):
        self.index_map = []
        for vi in range(len(self.volumes)):
            for _ in range(self.patches_per_volume):
                self.index_map.append(vi)
        random.shuffle(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def _load_volume(self, vol_idx: int):
        vol = self.volumes[vol_idx]
        # torch.load (trusted dataset) -> could be large; perform in worker
        image = torch.load(vol.image_path)  # shape typically [H,W,D,C] per your dataset
        mask = torch.load(vol.mask_path)
        # convert to tensor if numpy array
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask)
        # Permute to [C,D,H,W]
        # handle [H,W,D,C] (your case 160,192,155,4) -> permute(3,2,0,1)
        if image.ndim == 4 and image.shape[-1] <= 8:
            # try assume last dim is channels
            image = image.permute(3, 2, 0, 1).contiguous()  # [C,D,H,W]
        elif image.ndim == 3:
            image = image.unsqueeze(0)  # [1,D,H,W] or [1,H,W,D]? assume [D,H,W]
            # if ambiguous, try to standardize: user dataset uses C last, so this path unlikely
        else:
            # fallback: move last axis to first
            image = image.permute(-1, *range(0, image.ndim-1)).contiguous()

        # masks: [H,W,D,1] -> to [1,D,H,W]
        if mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.permute(3, 2, 0, 1).contiguous()
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.float()

        image = image.float()
        mask = mask.float()
        return image, mask

    def __getitem__(self, idx):
        vol_idx = self.index_map[idx]
        image, mask = self._load_volume(vol_idx)  # [C,D,H,W], [1,D,H,W]
        C, D, H, W = image.shape
        pd, ph, pw = self.patch_size
        # start indices (clamp start to 0 if dimension == patch)
        d0 = 0 if D == pd else random.randint(0, max(0, D - pd))
        h0 = 0 if H == ph else random.randint(0, max(0, H - ph))
        w0 = 0 if W == pw else random.randint(0, max(0, W - pw))
        img_patch = image[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]
        msk_patch = mask[:, d0:d0+pd, h0:h0+ph, w0:w0+pw]
        return img_patch, msk_patch

# ----------------------------
# Lightweight 3D model with transformer bottleneck
# (intentionally small to fit low-VRAM GPUs)
# ----------------------------
class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU()
        )
    def forward(self, x): return self.net(x)

class TinyTransformerBottleneck(nn.Module):
    def __init__(self, channels, embed_dim=64, heads=4, layers=1, patch_size=(2,4,4)):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        pz, py, px = patch_size
        self.patch_vol = pz*py*px
        self.proj = nn.Linear(channels*self.patch_vol, embed_dim)
        encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=layers)
        self.unproj = nn.Linear(embed_dim, channels*self.patch_vol)
        self.pos = None

    def forward(self, x):
        B, C, D, H, W = x.shape
        pz, py, px = self.patch_size
        # ensure divisibility
        assert D % pz == 0 and H % py == 0 and W % px == 0, "Input dims must be divisible by transformer patch_size"
        nz, ny, nx = D//pz, H//py, W//px
        patches = x.unfold(2, pz, pz).unfold(3, py, py).unfold(4, px, px)  # [B,C,nz,ny,nx,pz,py,px]
        patches = patches.permute(0,2,3,4,1,5,6,7).contiguous()  # [B,nz,ny,nx,C,pz,py,px]
        N = nz*ny*nx
        vec = patches.view(B, N, C*self.patch_vol)
        tokens = self.proj(vec)
        if (self.pos is None) or (self.pos.shape[1] != N):
            self.pos = nn.Parameter(torch.zeros(1, N, tokens.shape[-1], device=x.device))
            nn.init.trunc_normal_(self.pos, std=0.02)
        tokens = tokens + self.pos
        tokens = self.transformer(tokens)
        out = self.unproj(tokens)
        out = out.view(B, nz, ny, nx, C, pz, py, px)
        out = out.permute(0,4,1,5,2,6,3,7).contiguous().view(B, C, D, H, W)
        return out

class ResearchModel3D(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, base_ch=8, embed_dim=64, heads=4, layers=1):
        super().__init__()
        self.enc1 = Conv3DBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = Conv3DBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool3d(2)
        self.bottleneck = Conv3DBlock(base_ch*2, base_ch*4)
        # choose transformer patch size that divides bottleneck dims (will depend on input patch)
        self.transformer = TinyTransformerBottleneck(channels=base_ch*4, embed_dim=embed_dim, heads=heads, layers=layers, patch_size=(2,4,4))
        self.up = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec1 = Conv3DBlock(base_ch*4, base_ch*2)
        self.up2 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec2 = Conv3DBlock(base_ch*2, base_ch)
        self.final = nn.Conv3d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # x: [B,C,D,H,W]
        e1 = self.enc1(x)                # [B,base,D,H,W]
        e2 = self.enc2(self.pool1(e1))   # down1
        b = self.bottleneck(self.pool2(e2))
        t = self.transformer(b)
        u = self.up(t)
        cat = torch.cat([u, e2], dim=1)
        d1 = self.dec1(cat)
        u2 = self.up2(d1)
        cat2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(cat2)
        out = self.final(d2)  # logits
        return out

# ----------------------------
# Train / Validate loops
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch_idx:int, total_epochs:int):
    model.train()
    losses = []
    dices = []
    ious = []
    pbar = tqdm(loader, desc=f"Train {epoch_idx}/{total_epochs}", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            logits = model(imgs)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        d = dice_from_logits(logits, masks)
        j = iou_from_logits(logits, masks)

        losses.append(loss.item())
        dices.append(d)
        ious.append(j)

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}", iou=f"{j:.4f}")

        # free memory promptly
        del imgs, masks, logits, loss
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_dice = float(np.mean(dices)) if dices else 0.0
    avg_iou = float(np.mean(ious)) if ious else 0.0
    return avg_loss, avg_dice, avg_iou

def validate_one_epoch(model, loader, criterion):
    model.eval()
    losses = []
    dices = []
    ious = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validate", leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            with autocast(enabled=USE_AMP):
                logits = model(imgs)
                loss = criterion(logits, masks)
            d = dice_from_logits(logits, masks)
            j = iou_from_logits(logits, masks)
            losses.append(loss.item()); dices.append(d); ious.append(j)
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}", iou=f"{j:.4f}")

            del imgs, masks, logits, loss
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_dice = float(np.mean(dices)) if dices else 0.0
    avg_iou = float(np.mean(ious)) if ious else 0.0
    return avg_loss, avg_dice, avg_iou

# ----------------------------
# Main
# ----------------------------
def main():
    logging.info("Starting research training run")
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except Exception:
        # if deleted or broken, create new
        mlflow.create_experiment(MLFLOW_EXPERIMENT)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    train_root = os.path.join(DATA_ROOT, TRAIN_SUBDIR)
    val_root = os.path.join(DATA_ROOT, VAL_SUBDIR)

    train_ds = PatchVolumeDataset(train_root, patches_per_volume=PATCHES_PER_VOLUME, patch_size=PATCH_SIZE)
    val_ds = PatchVolumeDataset(val_root, patches_per_volume=1, patch_size=PATCH_SIZE) if os.path.exists(val_root) else None

    if len(train_ds) == 0:
        raise RuntimeError(f"No training volumes found under {train_root}")

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": NUM_WORKERS,
        "pin_memory": (DEVICE.type == 'cuda'),
        "persistent_workers": (NUM_WORKERS > 0)
    }
    train_loader = DataLoader(train_ds, **loader_kwargs)

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type=='cuda'), persistent_workers=(NUM_WORKERS>0))

    model = ResearchModel3D(in_ch=4, out_ch=1, base_ch=8, embed_dim=64, heads=4, layers=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # GradScaler for older PyTorch: do not pass device_type kwarg
    scaler = GradScaler(enabled=(USE_AMP and DEVICE.type=='cuda'))

    with mlflow.start_run():
        mlflow.log_params({
            "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patches_per_volume": PATCHES_PER_VOLUME,
            "use_amp": USE_AMP,
            "device": str(DEVICE),
            "num_workers": NUM_WORKERS
        })

        best_metric = -1.0
        patience = 8
        no_improve = 0

        for epoch in range(1, EPOCHS+1):
            t0 = time.time()
            train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, EPOCHS)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_dice", train_dice, step=epoch)
            mlflow.log_metric("train_iou", train_iou, step=epoch)
            logging.info(f"[Epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_iou={train_iou:.4f}")

            if val_loader is not None:
                val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_dice", val_dice, step=epoch)
                mlflow.log_metric("val_iou", val_iou, step=epoch)
                logging.info(f"[Epoch {epoch}/{EPOCHS}] val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}")
            else:
                val_loss, val_dice, val_iou = train_loss, train_dice, train_iou

            metric = val_dice if val_loader is not None else train_dice
            if metric > best_metric:
                best_metric = metric
                no_improve = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                mlflow.log_artifact(BEST_MODEL_PATH, artifact_path="models")
                logging.info(f"Saved best model to {BEST_MODEL_PATH} (metric={best_metric:.4f})")
            else:
                no_improve += 1

            # per-epoch checkpoint (optional)
            epoch_path = os.path.join(MODEL_DIR, f"research_model_epoch{epoch}.pt")
            torch.save(model.state_dict(), epoch_path)
            mlflow.log_artifact(epoch_path, artifact_path="models/epochs")

            logging.info(f"Epoch {epoch} finished in {(time.time()-t0)/60:.2f} minutes")

            if no_improve >= patience:
                logging.info(f"Early stopping: no improvement in {no_improve} epochs")
                break

        # export small TorchScript lite model on CPU to avoid GPU memory costs
        try:
            model_cpu = model.to('cpu').eval()
            pd, ph, pw = PATCH_SIZE
            dummy = torch.randn(1, 4, pd, ph, pw)
            traced = torch.jit.trace(model_cpu, dummy)
            traced.save(LITE_MODEL_PATH)
            mlflow.log_artifact(LITE_MODEL_PATH, artifact_path="models")
            logging.info(f"Saved lite model to {LITE_MODEL_PATH}")
        except Exception as e:
            logging.exception(f"Failed to create TorchScript lite: {e}")

    logging.info("Training complete.")

if __name__ == "__main__":
    main()

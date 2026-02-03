#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torchvision import models


def find_mask_path(mask_root: Path, patient_id: str) -> Optional[Path]:
    # Search for a mask file that includes the patient_id token in its path
    tok = patient_id.upper()
    candidates: List[Path] = []
    if not mask_root.exists():
        return None
    for p in mask_root.rglob("*.nii*"):
        up = str(p).upper()
        if tok in up:
            candidates.append(p)
    # Prefer names with stv/lesion/label/seg
    if not candidates:
        return None
    for p in candidates:
        name = p.name.lower()
        if any(k in name for k in ["stv", "lesion", "label", "seg"]):
            return p
    return candidates[0]


def load_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data


def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / (s + eps)


class NiftiSliceSegDataset(Dataset):
    def __init__(self, data_root: Path, mask_root: Path, split_file: Path, sequences: List[str], include_empty: bool = False):
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.channels = len(sequences)
        pids = [ln.strip() for ln in split_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for pid in pids:
            img_dir = data_root / "nifti" / pid
            if not img_dir.exists():
                continue
            # Load requested sequences if present; fallback to any found when missing
            imgs: List[np.ndarray] = []
            for seq in sequences:
                p = img_dir / f"{seq}.nii.gz"
                if p.exists():
                    imgs.append(load_nii(p))
            if not imgs:
                # fallback: any nii.gz
                found = sorted(img_dir.glob("*.nii.gz"))
                if not found:
                    continue
                imgs.append(load_nii(found[0]))
            # Stack channels; ensure same shape
            shapes = {im.shape for im in imgs}
            if len(shapes) != 1:
                # skip if shapes mismatch
                continue
            vol = np.stack(imgs, axis=0)  # (C, X, Y, Z)
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
            # normalize each channel
            for c in range(vol.shape[0]):
                vol[c] = zscore(vol[c])
            # Load mask
            mpath = (mask_root / pid / "lesion.nii.gz")
            if not mpath.exists():
                # heuristic search in mask_root
                hp = find_mask_path(mask_root, pid)
                if hp:
                    mpath = hp
            if not mpath.exists():
                continue
            mask = load_nii(mpath).astype(np.float32)
            # Validate shape
            if mask.shape != vol.shape[1:]:
                # skip mismatched cases for this simple 2D trainer
                continue
            # Build slices (axial by default: last axis Z)
            for z in range(mask.shape[2]):
                m2d = mask[:, :, z]
                if not include_empty and m2d.max() <= 0:
                    continue
                x2d = vol[:, :, :, z]  # (C, X, Y)
                self.samples.append((x2d, m2d))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        # to tensors (C, H, W) and (1, H, W)
        x_t = torch.from_numpy(x.astype(np.float32))
        y_t = torch.from_numpy((y > 0).astype(np.float32)).unsqueeze(0)
        return x_t, y_t


class DenseNetSeg(nn.Module):
    def __init__(self, in_channels: int, pretrained: bool = False):
        super().__init__()
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        # replace first conv to accept in_channels
        # backbone.features.conv0: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv0 = backbone.features.conv0
        new_conv0 = nn.Conv2d(in_channels, conv0.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained and in_channels == 3:
            new_conv0.weight.data.copy_(conv0.weight.data)
        else:
            nn.init.kaiming_normal_(new_conv0.weight, nonlinearity="relu")
        backbone.features.conv0 = new_conv0
        self.encoder = backbone.features  # outputs (N, 1024, H/32, W/32)

        # Simple upsampling decoder to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)  # (N, 1024, H/32, W/32)
        y = self.decoder(f)
        # Ensure output spatial matches input
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y


def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # pred, target: (N,1,H,W) logits vs binary
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    denom = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (2 * inter / denom).mean()


def train_one_epoch(model, loader, optim, scaler, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    bce = nn.BCEWithLogitsLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x)
            loss = bce(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        total_dice += dice_coef(logits.detach(), y).item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / max(n, 1), total_dice / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    bce = nn.BCEWithLogitsLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = bce(logits, y)
        total_loss += loss.item() * x.size(0)
        total_dice += dice_coef(logits, y).item() * x.size(0)
    n = len(loader.dataset)
    return total_loss / max(n, 1), total_dice / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Root with nifti/, lists/ under it")
    ap.add_argument("--mask-root", required=True, help="Root with STV masks (per-patient subfolders or heuristic)")
    ap.add_argument("--sequences", default="Pre", help="Comma-separated sequences to use as channels (e.g., Pre,Post_1,Post_2)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", required=True, help="Directory to save checkpoints and logs")
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    mask_root = Path(args.mask_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    train_split = data_root / "lists" / "train.txt"
    val_split = data_root / "lists" / "val.txt"
    if not train_split.exists() or not val_split.exists():
        raise FileNotFoundError(f"Missing split files under {data_root}/lists")

    train_ds = NiftiSliceSegDataset(data_root, mask_root, train_split, sequences=seqs, include_empty=False)
    val_ds = NiftiSliceSegDataset(data_root, mask_root, val_split, sequences=seqs, include_empty=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = DenseNetSeg(in_channels=len(seqs), pretrained=args.pretrained).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, train_loader, optim, scaler, device)
        va_loss, va_dice = eval_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | val loss {va_loss:.4f} dice {va_dice:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": va_dice}, out_dir / "last.ckpt")
        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": va_dice}, out_dir / "best.ckpt")


if __name__ == "__main__":
    main()






"""
Train a pretrained ResNet-50 on ODELIA_by_label_flat_Post2 (flat label folders with NIfTI volumes).

Expected source structure:
source_dir/
  benign/
    *.nii.gz
  malignant/
    *.nii.gz
  ... (any number of classes)

This script:
- Scans class folders and builds (image_path, class_index)
- Splits into train/val (stratified)
- Loads NIfTI volumes and extracts a representative slice (middle slice by default)
- Fine-tunes an ImageNet-pretrained ResNet-50 for classification
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Reuse model from existing script for consistency
from train_resnet50 import ResNet50Classifier


def find_odelia_files(source_dir: Path) -> Tuple[List[str], List[int], List[str]]:
    """
    Scan `source_dir` for class subfolders containing NIfTI files.
    Returns (paths, labels, class_names).
    """
    class_dirs = [d for d in sorted(source_dir.iterdir()) if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found in {source_dir}")
    class_names = [d.name for d in class_dirs]
    paths: List[str] = []
    labels: List[int] = []
    exts = {".nii", ".nii.gz", ".NII", ".NII.GZ"}
    for class_idx, class_dir in enumerate(class_dirs):
        for p in class_dir.iterdir():
            if p.is_file() and (p.suffix in exts or "".join(p.suffixes) in exts):
                paths.append(str(p))
                labels.append(class_idx)
    if not paths:
        raise ValueError(f"No NIfTI files found under {source_dir}")
    return paths, labels, class_names


class OdeliaFlatDataset(Dataset):
    """
    Dataset that loads NIfTI volumes and extracts a single 2D slice per sample.
    """
    def __init__(self, paths: List[str], labels: List[int], transform=None, input_channels: int = 1, slice_strategy: str = "middle"):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.input_channels = input_channels
        self.slice_strategy = slice_strategy

    def __len__(self) -> int:
        return len(self.paths)

    def _load_nifti_slice(self, path: str) -> Image.Image:
        import nibabel as nib
        nii = nib.load(path)
        vol = nii.get_fdata(dtype=np.float32)
        if vol.ndim == 4:
            # If time dimension exists, take first volume
            vol = vol[..., 0]
        z = vol.shape[2] // 2 if self.slice_strategy == "middle" else int(np.argmax(vol.sum(axis=(0, 1))))
        img2d = vol[:, :, z]
        # Normalize robustly to [0,255]
        finite_vals = img2d[np.isfinite(img2d)]
        if finite_vals.size == 0:
            img2d = np.zeros_like(img2d, dtype=np.float32)
        else:
            lo, hi = np.percentile(finite_vals, [2.0, 98.0])
            if hi <= lo:
                hi = lo + 1.0
            img2d = np.clip((img2d - lo) / (hi - lo), 0.0, 1.0)
        img_uint8 = (img2d * 255).astype(np.uint8)
        return Image.fromarray(img_uint8, mode="L")

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        label = self.labels[idx]
        img = self._load_nifti_slice(path)
        if self.input_channels == 3:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def plot_training_history(history: dict, save_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0, 0].plot(epochs, history['train_loss'], label='Train'); axes[0, 0].plot(epochs, history['val_loss'], label='Val'); axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(epochs, history['train_accuracy'], label='Train'); axes[0, 1].plot(epochs, history['val_accuracy'], label='Val'); axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[0, 2].plot(epochs, history['train_precision'], label='Train'); axes[0, 2].plot(epochs, history['val_precision'], label='Val'); axes[0, 2].set_title('Precision'); axes[0, 2].legend(); axes[0, 2].grid(True)
    axes[1, 0].plot(epochs, history['train_recall'], label='Train'); axes[1, 0].plot(epochs, history['val_recall'], label='Val'); axes[1, 0].set_title('Recall'); axes[1, 0].legend(); axes[1, 0].grid(True)
    axes[1, 1].plot(epochs, history['train_f1'], label='Train'); axes[1, 1].plot(epochs, history['val_f1'], label='Val'); axes[1, 1].set_title('F1'); axes[1, 1].legend(); axes[1, 1].grid(True)
    if 'lr' in history: axes[1, 2].plot(epochs, history['lr'], label='LR'); axes[1, 2].set_yscale('log'); axes[1, 2].legend(); axes[1, 2].grid(True)
    else: axes[1, 2].axis('off')
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    epoch_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {'loss': epoch_loss, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(dataloader, desc='[Val]'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    epoch_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {'loss': epoch_loss, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on ODELIA_by_label_flat_Post2")
    parser.add_argument('--source-dir', type=str, required=True, help='Path to ODELIA_by_label_flat_Post2')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation fraction')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--input-channels', type=int, default=1, choices=[1, 3], help='1=grayscale, 3=RGB')
    parser.add_argument('--image-size', type=int, default=224, help='Resize square')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'step', 'none'])
    parser.add_argument('--step-size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='./outputs_resnet50_odelia')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--class-weights', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f: json.dump(vars(args), f, indent=4)

    # Data
    paths, labels, class_names = find_odelia_files(Path(args.source_dir))
    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels, test_size=args.val_ratio, random_state=args.random_seed, stratify=labels
    )

    # Transforms
    mean_std_gray = (0.485,), (0.229,)
    mean_std_rgb = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean, std = (mean_std_gray if args.input_channels == 1 else mean_std_rgb)
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_ds = OdeliaFlatDataset(X_train, y_train, transform=train_transform, input_channels=args.input_channels)
    val_ds = OdeliaFlatDataset(X_val, y_val, transform=val_transform, input_channels=args.input_channels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # Model
    num_classes = len(class_names)
    model = ResNet50Classifier(num_classes=num_classes, pretrained=args.pretrained, input_channels=args.input_channels).to(device)

    # Loss
    if args.class_weights:
        # compute weights from train set
        class_counts = np.bincount(y_train, minlength=num_classes)
        weights = (class_counts.sum() / (len(class_counts) * class_counts.clip(min=1))).astype(np.float32)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == 'plateau':
        # 'verbose' is not available in some torch versions; omit for compatibility
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    history = {k: [] for k in ['train_loss','train_accuracy','train_precision','train_recall','train_f1','val_loss','val_accuracy','val_precision','val_recall','val_f1','lr']}
    best_val_loss = float('inf'); best_val_f1 = 0.0; patience_counter = 0

    print(f'Classes: {class_names}')
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Num classes: {num_classes}')
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}\n' + '-'*50)
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        if scheduler:
            if args.scheduler == 'plateau': scheduler.step(val_metrics['loss'])
            else: scheduler.step()
        for key in ['loss','accuracy','precision','recall','f1']:
            history[f'train_{key}'].append(train_metrics[key]); history[f'val_{key}'].append(val_metrics[key])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        print(f"Train - Loss {train_metrics['loss']:.4f} Acc {train_metrics['accuracy']:.4f} F1 {train_metrics['f1']:.4f}")
        print(f"Val   - Loss {val_metrics['loss']:.4f} Acc {val_metrics['accuracy']:.4f} F1 {val_metrics['f1']:.4f}")
        # Save bests
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']; patience_counter = 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_metrics['loss'], 'val_f1': val_metrics['f1'], 'history': history}, output_dir / 'best_model.pt')
            print('✓ Saved best model (lowest val loss)')
        elif val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_metrics['loss'], 'val_f1': val_metrics['f1'], 'history': history}, output_dir / 'best_f1_model.pt')
            print('✓ Saved best model (highest val F1)')
        patience_counter += 1
        # Optional manual early stopping could be added if desired
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'history': history}, output_dir / f'checkpoint_epoch_{epoch}.pt')

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'history': history}, output_dir / 'final_model.pt')
    plot_training_history(history, output_dir / 'training_history.png')
    with open(output_dir / 'training_history.json', 'w') as f: json.dump(history, f, indent=4)
    print(f"\nTraining complete. Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()


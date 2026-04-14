#!/usr/bin/env python3
import csv
import numpy as np
import nibabel as nib
from pathlib import Path

# =========================
# Paths
# =========================
PRED_ROOT = Path("/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/predsTs_ISPY1")
GT_DIR    = Path("/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/nnUNet_raw/Dataset503_ISPY1/labelsTs")
OUT_DIR   = Path("/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/eval_ISPY1")

# =========================
# Metric functions
# =========================
def dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)

def iou_score(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return intersection / (union + 1e-8)

def precision_score(pred, gt):
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    return tp / (tp + fp + 1e-8)

def recall_score(pred, gt):
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    return tp / (tp + fn + 1e-8)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted([p for p in PRED_ROOT.iterdir() if p.is_dir()])
    if not model_dirs:
        raise RuntimeError(f"No model subfolders found in {PRED_ROOT}")

    for model_dir in model_dirs:
        out_file = OUT_DIR / f"metrics_{model_dir.name}.csv"
        pred_files = sorted(model_dir.glob("*.nii*"))

        if len(pred_files) == 0:
            print(f"[WARNING] No prediction files found in {model_dir}, skipping.")
            continue

        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case", "dice", "iou", "precision", "recall"])

            for pred_path in pred_files:
                case_name = pred_path.name
                gt_path = GT_DIR / case_name

                if not gt_path.exists():
                    print(f"[WARNING] Missing GT for {case_name} (model {model_dir.name}), skipping.")
                    continue

                pred = nib.load(pred_path).get_fdata()
                gt   = nib.load(gt_path).get_fdata()

                # Binarize (assumes foreground label > 0)
                pred = (pred > 0).astype(np.uint8)
                gt   = (gt > 0).astype(np.uint8)

                dice = dice_score(pred, gt)
                iou  = iou_score(pred, gt)
                prec = precision_score(pred, gt)
                rec  = recall_score(pred, gt)

                writer.writerow([case_name, dice, iou, prec, rec])

        print(f"[DONE] {model_dir.name}: wrote {out_file}")

if __name__ == "__main__":
    main()
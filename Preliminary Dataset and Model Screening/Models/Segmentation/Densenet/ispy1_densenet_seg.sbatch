#!/bin/bash
#SBATCH --job-name=densenet-ispy1
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO=/arc/project/st-ilker-1/SBME-Capstone-2025-Breast-Lesion-MRI-Analysis
DATA_ROOT=/scratch/st-ilker-1/$USER/ispy1
MASK_ROOT=/scratch/st-ilker-1/$USER/ispy1/stv_manual_masks
OUT_DIR=/scratch/st-ilker-1/$USER/ispy1_runs/densenet_seg
SEQS="Pre"   # or "Pre,Post_1,Post_2"

mkdir -p "$OUT_DIR" "$(dirname "$REPO")" "$DATA_ROOT" "$MASK_ROOT"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispy1

python "$REPO/Preliminary Dataset and Model Screening/Models/Segmentation/Densenet/train_densenet_seg.py" \
  --data-root "$DATA_ROOT" \
  --mask-root "$MASK_ROOT" \
  --sequences "$SEQS" \
  --epochs 20 \
  --batch-size 8 \
  --out-dir "$OUT_DIR" \
  --pretrained

echo "Done."






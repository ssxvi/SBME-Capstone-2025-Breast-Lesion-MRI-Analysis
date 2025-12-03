#!/bin/bash
#SBATCH --job-name=ispy1-prepare
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Purpose:
#  - Run conversion/reorientation/resampling/splits using already-downloaded DICOM in /project
#  - No internet required on compute nodes. If you need download, run prepare_ispy1.py on login/DTN first.

set -euo pipefail

# Edit these paths
OUT_ROOT=/scratch/st-ilker-1/$USER/ispy1
STV_ROOT=/scratch/st-ilker-1/$USER/ispy1_seg_nifti   # optional; leave as-is if not available
MAX_PATIENTS=50   # limit total patients for quick pass; leave empty for all
VAL_FRAC=0.1
TEST_FRAC=0.1

# Environment setup
module purge
# If available on your cluster:
# module load dcm2niix

# Conda env (recommended)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate ispy1 || true
fi

mkdir -p "$OUT_ROOT/logs"

CMD=(python "Preliminary Dataset and Model Screening/Dataset Access/ISPY1/prepare_ispy1.py"
  --out-root "$OUT_ROOT"
  --val-frac "$VAL_FRAC"
  --test-frac "$TEST_FRAC"
  --skip-download           # assume downloads already done on login/DTN
)

if [ -n "${STV_ROOT:-}" ] && [ -d "$STV_ROOT" ]; then
  CMD+=(--stv-root "$STV_ROOT")
fi
if [ -n "${MAX_PATIENTS:-}" ]; then
  CMD+=(--max-patients "$MAX_PATIENTS")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done."



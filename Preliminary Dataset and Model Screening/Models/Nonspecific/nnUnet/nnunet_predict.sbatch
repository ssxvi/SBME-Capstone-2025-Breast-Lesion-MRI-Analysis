#!/bin/bash
#SBATCH --job-name=nnunet_pred_allModels
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=nnunet_predict_%j.out
#SBATCH --error=nnunet_predict_%j.err

source /home/yfeng40/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/../nnUNet/conda_envs/nnunet2_gpu 2>/dev/null || \
conda activate /scratch/st-ilker-1/yfeng40/nnUNet/conda_envs/nnunet2_gpu

# nnU-Net paths
export nnUNet_raw=/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/nnUNet_results

# Matplotlib / font cache fixes (prevents permission warnings)
export XDG_CACHE_HOME=/scratch/st-ilker-1/yfeng40/.cache
export MPLCONFIGDIR=/scratch/st-ilker-1/yfeng40/.cache/matplotlib
mkdir -p "$XDG_CACHE_HOME/fontconfig" "$MPLCONFIGDIR"

DATASET=Dataset503_ISPY1
PLANS=nnUNetPlans
FOLD=0
CHK=checkpoint_final.pth

declare -a CFGS=("3d_fullres" "3d_fullres" "3d_fullres" "2d")
declare -a TRAINERS=("nnUNetTrainer" "nnUNetTrainer_CEHeavy" "nnUNetTrainer_DiceHeavy" "nnUNetTrainer")
declare -a TAGS=("baseline3d" "CEHeavy3d" "DiceHeavy3d" "baseline2d")

for idx in ${!TRAINERS[@]}; do
  CFG="${CFGS[$idx]}"
  TRAINER="${TRAINERS[$idx]}"
  TAG="${TAGS[$idx]}"

  MODEL_DIR="$nnUNet_results/$DATASET/${TRAINER}__${PLANS}__${CFG}"
  OUTDIR="/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/predsTs_ISPY1/${TAG}"

  echo "=== Predicting: $TRAINER / $CFG -> $OUTDIR ==="
  ls -ld "$MODEL_DIR" || exit 1
  mkdir -p "$OUTDIR"

  nnUNetv2_predict \
    -d "$DATASET" \
    -c "$CFG" \
    -f "$FOLD" \
    -tr "$TRAINER" \
    -p "$PLANS" \
    -chk "$CHK" \
    -i "$nnUNet_raw/$DATASET/imagesTs" \
    -o "$OUTDIR"
done

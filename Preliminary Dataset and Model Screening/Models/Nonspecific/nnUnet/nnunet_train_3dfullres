#!/bin/bash
#SBATCH --job-name=nnunet503_3d_fullres_f0
#SBATCH --account=st-ilker-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -eo pipefail

# ---- make logs dir ----
mkdir -p logs

# ---- activate env ----
source ~/.bashrc
conda activate /scratch/st-ilker-1/yfeng40/nnUNet/conda_envs/nnunet2_gpu

# ---- nnU-Net paths ----
export nnUNet_raw="/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/nnUNet_results"

# ---- fix matplotlib/fontconfig cache warnings on HPC ----
export TMPDIR=${SLURM_TMPDIR:-/tmp}
mkdir -p $TMPDIR/mplconfig $TMPDIR/xdg_cache $TMPDIR/fontconfig
export MPLCONFIGDIR=$TMPDIR/mplconfig
export XDG_CACHE_HOME=$TMPDIR/xdg_cache
export FC_CACHEDIR=$TMPDIR/fontconfig

# ---- writable cache dirs for torch.compile / triton / inductor ----
export TMPDIR=${SLURM_TMPDIR:-/tmp}
mkdir -p $TMPDIR/triton $TMPDIR/torchinductor $TMPDIR/torch_extensions $TMPDIR/xdg_cache $TMPDIR/mplconfig $TMPDIR/fontconfig

export TRITON_CACHE_DIR=$TMPDIR/triton
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_EXTENSIONS_DIR=$TMPDIR/torch_extensions
export XDG_CACHE_HOME=$TMPDIR/xdg_cache
export MPLCONFIGDIR=$TMPDIR/mplconfig
export FC_CACHEDIR=$TMPDIR/fontconfig

# ---- optional: sanity checks ----
which nnUNetv2_train
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# ---- train (IMPORTANT: use -device, not --device) ----
nnUNetv2_train 503 3d_fullres 0 -device cuda


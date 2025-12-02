#!/bin/bash
#SBATCH --job-name=resnet_train
#SBATCH --account=def-<your_account>  # Replace with your Sockeye account
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# ResNet Training Job for Sockeye
# Trains ResNet50 on TCIA (ISPY2 + DUKE) breast MRI data

# Load required modules
module load python/3.10
module load cuda/11.8  # Adjust CUDA version as needed
module load cudnn/8.7.0  # Adjust cuDNN version as needed

# Set up environment
export PROJECT_DIR="${HOME}/projects/breast-lesion-mri"
export DATA_DIR="${SCRATCH}/tcia_processed/combined"  # Use combined dataset
export OUTPUT_DIR="${SCRATCH}/resnet_outputs"
export VENV_DIR="${PROJECT_DIR}/venv"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

# Set CUDA device (should be automatic, but just in case)
export CUDA_VISIBLE_DEVICES=0

# Print GPU information
echo "=========================================="
echo "GPU Information:"
echo "=========================================="
nvidia-smi

# Run training
echo "=========================================="
echo "Starting ResNet50 training..."
echo "=========================================="
cd "${PROJECT_DIR}/Preliminary Dataset and Model Screening/Models/Classification/Resnet"

python train_resnet50.py \
    --data-dir "${DATA_DIR}" \
    --input-channels 1 \
    --image-size 224 \
    --pretrained \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --scheduler plateau \
    --num-workers 8 \
    --output-dir "${OUTPUT_DIR}/tcia_resnet50_$(date +%Y%m%d_%H%M%S)" \
    --save-interval 10 \
    --early-stopping-patience 20 \
    --class-weights \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log

echo "=========================================="
echo "Training complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="


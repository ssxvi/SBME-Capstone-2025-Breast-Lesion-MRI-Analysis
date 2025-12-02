#!/bin/bash
#SBATCH --job-name=tcia_process
#SBATCH --account=def-<your_account>  # Replace with your Sockeye account
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/process_%j.log
#SBATCH --error=logs/process_%j.err

# TCIA Data Processing Job for Sockeye
# Converts DICOM to NIfTI and organizes for training

# Load required modules
module load python/3.10
module load gcc/9.3.0  # May be needed for some Python packages

# Set up environment
export PROJECT_DIR="${HOME}/projects/breast-lesion-mri"
export DATA_DIR="${SCRATCH}/tcia_data"
export PROCESSED_DIR="${SCRATCH}/tcia_processed"
export VENV_DIR="${PROJECT_DIR}/venv"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Create directories
mkdir -p "${PROCESSED_DIR}/ispy2"
mkdir -p "${PROCESSED_DIR}/duke"
mkdir -p "${PROCESSED_DIR}/combined"  # Combined dataset
mkdir -p logs

# Process ISPY2 dataset
echo "=========================================="
echo "Processing ISPY2 dataset..."
echo "=========================================="
cd "${PROJECT_DIR}/Preliminary Dataset and Model Screening/Dataset Access/TCIA"
python process_tcia_dicom.py \
    --input-dir "${DATA_DIR}/ispy2" \
    --output-dir "${PROCESSED_DIR}/ispy2" \
    --metadata "${DATA_DIR}/ispy2/metadata.csv" \
    --train-ratio 0.8 \
    2>&1 | tee logs/ispy2_process_$(date +%Y%m%d_%H%M%S).log

# Process DUKE dataset
echo "=========================================="
echo "Processing DUKE dataset..."
echo "=========================================="
python process_tcia_dicom.py \
    --input-dir "${DATA_DIR}/duke" \
    --output-dir "${PROCESSED_DIR}/duke" \
    --metadata "${DATA_DIR}/duke/metadata.csv" \
    --train-ratio 0.8 \
    2>&1 | tee logs/duke_process_$(date +%Y%m%d_%H%M%S).log

# Optionally combine datasets
echo "=========================================="
echo "Combining ISPY2 and DUKE datasets..."
echo "=========================================="
mkdir -p "${PROCESSED_DIR}/combined/train/class0"
mkdir -p "${PROCESSED_DIR}/combined/train/class1"
mkdir -p "${PROCESSED_DIR}/combined/val/class0"
mkdir -p "${PROCESSED_DIR}/combined/val/class1"

# Copy files from both datasets to combined directory
for dataset in ispy2 duke; do
    for split in train val; do
        for class in class0 class1; do
            if [ -d "${PROCESSED_DIR}/${dataset}/${split}/${class}" ]; then
                cp -r "${PROCESSED_DIR}/${dataset}/${split}/${class}"/* \
                      "${PROCESSED_DIR}/combined/${split}/${class}/" 2>/dev/null || true
            fi
        done
    done
done

echo "=========================================="
echo "Processing complete!"
echo "ISPY2 processed: ${PROCESSED_DIR}/ispy2"
echo "DUKE processed: ${PROCESSED_DIR}/duke"
echo "Combined dataset: ${PROCESSED_DIR}/combined"
echo "=========================================="


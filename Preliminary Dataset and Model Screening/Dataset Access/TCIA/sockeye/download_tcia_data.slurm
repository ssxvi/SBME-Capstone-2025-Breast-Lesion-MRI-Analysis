#!/bin/bash
#SBATCH --job-name=tcia_download
#SBATCH --account=def-tkliu03  # Replace with your Sockeye account
#SBATCH --time=12:00:00  # Full dataset download can take 6-12 hours
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/download_%j.log
#SBATCH --error=logs/download_%j.err

# TCIA Data Download Job for Sockeye
# Downloads ISPY2 and DUKE datasets from TCIA using NBIA Data Retriever

# Load required modules
module load python/3.10
module load java  # NBIA Data Retriever may require Java

# Set up environment
export PROJECT_DIR="${HOME}/projects/breast-lesion-mri"
export DATA_DIR="${SCRATCH}/tcia_data"  # Use SCRATCH for large datasets
export VENV_DIR="${PROJECT_DIR}/venv"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Create directories
mkdir -p "${DATA_DIR}/ispy2"
mkdir -p "${DATA_DIR}/duke"
mkdir -p logs

# TCIA credentials (optional - not required as of July 2025)
# Only needed for legacy controlled-access collections
# If needed, set via: export TCIA_USERNAME="your_username" and export TCIA_PASSWORD="your_password"
export TCIA_USERNAME="${TCIA_USERNAME:-}"
export TCIA_PASSWORD="${TCIA_PASSWORD:-}"

# Download ISPY2 dataset
echo "=========================================="
echo "Downloading ISPY2 dataset..."
echo "=========================================="
cd "${PROJECT_DIR}/Preliminary Dataset and Model Screening/Dataset Access/ISPY2"

# Build command with optional credentials
if [ -n "${TCIA_USERNAME}" ] && [ -n "${TCIA_PASSWORD}" ]; then
    python download-ispy2.py \
        --username "${TCIA_USERNAME}" \
        --password "${TCIA_PASSWORD}" \
        --output-dir "${DATA_DIR}/ispy2" \
        2>&1 | tee logs/ispy2_download_$(date +%Y%m%d_%H%M%S).log
else
    echo "Note: Downloading without credentials (not required as of July 2025)"
    python download-ispy2.py \
        --output-dir "${DATA_DIR}/ispy2" \
        2>&1 | tee logs/ispy2_download_$(date +%Y%m%d_%H%M%S).log
fi

# Download DUKE dataset
echo "=========================================="
echo "Downloading DUKE dataset..."
echo "=========================================="
cd "${PROJECT_DIR}/Preliminary Dataset and Model Screening/Dataset Access/Duke"

# Build command with optional credentials
if [ -n "${TCIA_USERNAME}" ] && [ -n "${TCIA_PASSWORD}" ]; then
    python duke-NBIA-streaming.py \
        --username "${TCIA_USERNAME}" \
        --password "${TCIA_PASSWORD}" \
        --output-dir "${DATA_DIR}/duke" \
        2>&1 | tee logs/duke_download_$(date +%Y%m%d_%H%M%S).log
else
    echo "Note: Downloading without credentials (not required as of July 2025)"
    python duke-NBIA-streaming.py \
        --output-dir "${DATA_DIR}/duke" \
        2>&1 | tee logs/duke_download_$(date +%Y%m%d_%H%M%S).log
fi

echo "=========================================="
echo "Download complete!"
echo "ISPY2 data: ${DATA_DIR}/ispy2"
echo "DUKE data: ${DATA_DIR}/duke"
echo "=========================================="


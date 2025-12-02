#!/bin/bash
#SBATCH --job-name=tcia_test
#SBATCH --account=def-tkliu03  # Replace with your Sockeye account
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/test_download_%j.log
#SBATCH --error=logs/test_download_%j.err

# Test Download Script for TCIA Data
# Downloads a small subset (5 patients each) to test the workflow

# Load required modules
module load python/3.10
module load java  # NBIA Data Retriever may require Java

# Set up environment
export PROJECT_DIR="${HOME}/projects/breast-lesion-mri"
export TEST_DATA_DIR="${SCRATCH}/tcia_test_data"  # Use SCRATCH for test data
export VENV_DIR="${PROJECT_DIR}/venv"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Create directories
mkdir -p "${TEST_DATA_DIR}"
mkdir -p logs

# TCIA credentials (optional - not required as of July 2025)
export TCIA_USERNAME="${TCIA_USERNAME:-}"
export TCIA_PASSWORD="${TCIA_PASSWORD:-}"

echo "=========================================="
echo "TCIA Test Subset Download"
echo "=========================================="
echo "This will download a small subset to test the workflow"
echo "Max patients per dataset: 5"
echo "Max time per dataset: 30 minutes"
echo "Output: ${TEST_DATA_DIR}"
echo "=========================================="

# Run test download script
cd "${PROJECT_DIR}/Preliminary Dataset and Model Screening/Dataset Access/TCIA"

# Build command with optional credentials
if [ -n "${TCIA_USERNAME}" ] && [ -n "${TCIA_PASSWORD}" ]; then
    python download_test_subset.py \
        --max-patients 5 \
        --max-time 30 \
        --output-dir "${TEST_DATA_DIR}" \
        --username "${TCIA_USERNAME}" \
        --password "${TCIA_PASSWORD}" \
        2>&1 | tee logs/test_download_$(date +%Y%m%d_%H%M%S).log
else
    echo "Note: Downloading without credentials (not required as of July 2025)"
    python download_test_subset.py \
        --max-patients 5 \
        --max-time 30 \
        --output-dir "${TEST_DATA_DIR}" \
        2>&1 | tee logs/test_download_$(date +%Y%m%d_%H%M%S).log
fi

echo ""
echo "=========================================="
echo "Test download complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Check downloaded data:"
echo "   ls -lh ${TEST_DATA_DIR}/ispy2"
echo "   ls -lh ${TEST_DATA_DIR}/duke"
echo ""
echo "2. Test processing (submit separately):"
echo "   sbatch process_test_data.slurm"
echo ""
echo "3. If successful, proceed with full download:"
echo "   sbatch download_tcia_data.slurm"
echo "=========================================="


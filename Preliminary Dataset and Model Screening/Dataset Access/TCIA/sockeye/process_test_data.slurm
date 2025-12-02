#!/bin/bash
#SBATCH --job-name=tcia_test_process
#SBATCH --account=def-tkliu03  # Replace with your Sockeye account
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/test_process_%j.log
#SBATCH --error=logs/test_process_%j.err

# Test Processing Script for TCIA Data
# Processes the test subset to verify the workflow

# Load required modules
module load python/3.10
module load gcc/9.3.0

# Set up environment
export PROJECT_DIR="${HOME}/projects/breast-lesion-mri"
export TEST_DATA_DIR="${SCRATCH}/tcia_test_data"
export TEST_PROCESSED_DIR="${SCRATCH}/tcia_test_processed"
export VENV_DIR="${PROJECT_DIR}/venv"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Create directories
mkdir -p "${TEST_PROCESSED_DIR}"
mkdir -p logs

echo "=========================================="
echo "Processing TCIA Test Data"
echo "=========================================="

cd "${PROJECT_DIR}/Preliminary Dataset and Model Screening/Dataset Access/TCIA"

# Process ISPY2 test data
if [ -d "${TEST_DATA_DIR}/ispy2" ] && [ "$(ls -A ${TEST_DATA_DIR}/ispy2 2>/dev/null)" ]; then
    echo "Processing ISPY2 test data..."
    python process_tcia_dicom.py \
        --input-dir "${TEST_DATA_DIR}/ispy2" \
        --output-dir "${TEST_PROCESSED_DIR}/ispy2" \
        --train-ratio 0.8 \
        2>&1 | tee logs/test_process_ispy2_$(date +%Y%m%d_%H%M%S).log
else
    echo "Warning: ISPY2 test data not found, skipping..."
fi

# Process DUKE test data
if [ -d "${TEST_DATA_DIR}/duke" ] && [ "$(ls -A ${TEST_DATA_DIR}/duke 2>/dev/null)" ]; then
    echo "Processing DUKE test data..."
    python process_tcia_dicom.py \
        --input-dir "${TEST_DATA_DIR}/duke" \
        --output-dir "${TEST_PROCESSED_DIR}/duke" \
        --train-ratio 0.8 \
        2>&1 | tee logs/test_process_duke_$(date +%Y%m%d_%H%M%S).log
else
    echo "Warning: DUKE test data not found, skipping..."
fi

echo ""
echo "=========================================="
echo "Test processing complete!"
echo "=========================================="
echo "Check processed data:"
echo "  ${TEST_PROCESSED_DIR}/ispy2"
echo "  ${TEST_PROCESSED_DIR}/duke"
echo ""
echo "If successful, you can proceed with full dataset download and processing."
echo "=========================================="


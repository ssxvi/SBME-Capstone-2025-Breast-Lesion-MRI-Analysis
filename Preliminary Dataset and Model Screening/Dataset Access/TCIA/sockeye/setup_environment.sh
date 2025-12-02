#!/bin/bash
# Setup script for Sockeye environment
# Run this once to set up your Python environment

set -e  # Exit on error

# Configuration
PROJECT_DIR="${HOME}/projects/breast-lesion-mri"
VENV_DIR="${PROJECT_DIR}/venv"
SCRATCH_DIR="${SCRATCH}/tcia_data"

echo "=========================================="
echo "Setting up Sockeye environment for TCIA training"
echo "=========================================="

# Load Python module
module load python/3.10

# Create project directory
mkdir -p "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv "${VENV_DIR}"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing Python packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib tqdm
pip install pydicom nibabel scipy
pip install pillow

# Create necessary directories
echo "Creating directories..."
mkdir -p "${SCRATCH_DIR}/ispy2"
mkdir -p "${SCRATCH_DIR}/duke"
mkdir -p "${SCRATCH_DIR}/processed"
mkdir -p "${SCRATCH_DIR}/outputs"
mkdir -p "${PROJECT_DIR}/logs"

# Download NBIA Data Retriever (if not already installed)
echo "=========================================="
echo "NBIA Data Retriever setup:"
echo "=========================================="
echo "The NBIA Data Retriever must be downloaded manually from:"
echo "https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images"
echo ""
echo "After downloading, extract it and add to PATH or specify path in scripts"
echo ""

# Set up TCIA credentials (optional - not required as of July 2025)
echo "=========================================="
echo "TCIA Credentials setup (OPTIONAL):"
echo "=========================================="
echo "As of July 2025, TCIA no longer requires account registration."
echo "Credentials are only needed for legacy controlled-access collections."
echo ""
echo "If needed, set credentials as environment variables:"
echo "  export TCIA_USERNAME='your_username'"
echo "  export TCIA_PASSWORD='your_password'"
echo ""
echo "Or add to your ~/.bashrc for persistence:"
echo "  echo 'export TCIA_USERNAME=\"your_username\"' >> ~/.bashrc"
echo "  echo 'export TCIA_PASSWORD=\"your_password\"' >> ~/.bashrc"
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Download NBIA Data Retriever if needed"
echo "2. (Optional) Set TCIA credentials if accessing legacy collections"
echo "3. Submit download job: sbatch download_tcia_data.slurm"
echo "4. After download, submit processing job: sbatch process_tcia_data.slurm"
echo "5. After processing, submit training job: sbatch train_resnet.slurm"
echo "=========================================="


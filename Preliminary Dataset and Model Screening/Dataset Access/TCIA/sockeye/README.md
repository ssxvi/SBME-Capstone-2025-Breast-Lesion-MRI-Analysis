# TCIA Training Setup for UBC Sockeye

This directory contains scripts and instructions for setting up training on UBC's Sockeye HPC cluster using ISPY2 and DUKE datasets from TCIA.

## Overview

The workflow consists of three main steps:
1. **Download**: Download ISPY2 and DUKE datasets from TCIA using NBIA Data Retriever
2. **Process**: Convert DICOM files to NIfTI format and organize into train/val splits
3. **Train**: Train ResNet50 model on the processed data

## Prerequisites

1. **Sockeye Account**: You need an active Sockeye account with appropriate allocations
2. **TCIA Account**: **NOT REQUIRED** - As of July 2025, TCIA no longer requires account registration for most datasets. Credentials are only needed for legacy controlled-access collections.
3. **NBIA Data Retriever**: Download from https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images

## Setup Instructions

### 1. Initial Setup

```bash
# Navigate to the sockeye directory
cd "Preliminary Dataset and Model Screening/Dataset Access/TCIA/sockeye"

# Run the setup script
bash setup_environment.sh
```

### 2. Configure Your Account

Edit the SLURM scripts to set your account:

```bash
# In each .slurm file, replace:
#SBATCH --account=def-<your_account>

# With your actual account, e.g.:
#SBATCH --account=def-username
```

### 3. Set TCIA Credentials (Optional)

**Note**: As of July 2025, TCIA no longer requires account registration for most datasets. You can skip this step unless you're accessing legacy controlled-access collections.

If needed for specific collections, set credentials as environment variables:

```bash
# For current session
export TCIA_USERNAME="your_username"
export TCIA_PASSWORD="your_password"

# Or add to ~/.bashrc for persistence
echo 'export TCIA_USERNAME="your_username"' >> ~/.bashrc
echo 'export TCIA_PASSWORD="your_password"' >> ~/.bashrc
source ~/.bashrc
```

### 4. Update Paths

Edit the SLURM scripts to match your project structure:

- `PROJECT_DIR`: Path to your project root
- `DATA_DIR`: Where to store downloaded data (use `$SCRATCH` for large datasets)
- `VENV_DIR`: Path to your virtual environment

## Usage

### Step 1: Download TCIA Data

```bash
# Submit download job
sbatch download_tcia_data.slurm

# Monitor job
squeue -u $USER

# Check logs
tail -f logs/download_*.log
```

**Note**: Downloading large datasets can take many hours. The ISPY2 and DUKE datasets are several GB in size.

### Step 2: Process DICOM Data

After downloads complete, process the DICOM files:

```bash
# Submit processing job
sbatch process_tcia_data.slurm

# Monitor job
squeue -u $USER

# Check logs
tail -f logs/process_*.log
```

**Note**: You may need to create metadata CSV files with patient labels. The script expects:
- Column: `PatientID` (patient identifier)
- Column: `Label` (0 for benign, 1 for malignant)

### Step 3: Train Model

After processing is complete, train the ResNet50 model:

```bash
# Submit training job
sbatch train_resnet.slurm

# Monitor job
squeue -u $USER

# Check logs
tail -f logs/train_*.log
```

## File Structure

```
sockeye/
├── setup_environment.sh      # Initial environment setup
├── download_tcia_data.slurm # Download job script
├── process_tcia_data.slurm  # Processing job script
├── train_resnet.slurm       # Training job script
├── README.md                # This file
└── logs/                    # Job logs (created automatically)
```

## Data Organization

After processing, data will be organized as:

```
processed_data/
├── train/
│   ├── class0/  (benign)
│   └── class1/  (malignant)
└── val/
    ├── class0/
    └── class1/
```

## Troubleshooting

### NBIA Data Retriever Not Found

If you get an error about NBIA Data Retriever:
1. Download it from the TCIA website
2. Extract it to a location accessible on Sockeye
3. Add the path to your scripts or add to PATH

### Out of Memory Errors

If you encounter memory errors:
- Reduce batch size in training script
- Process datasets separately instead of combining
- Request more memory in SLURM script: `#SBATCH --mem=64G`

### GPU Not Available

If GPU is not available:
- Check GPU availability: `squeue -u $USER`
- Verify GPU request: `#SBATCH --gres=gpu:1`
- Check CUDA module: `module load cuda/11.8`

### DICOM Processing Errors

If DICOM processing fails:
- Verify DICOM files are valid
- Check metadata CSV format
- Try processing with `--full-volume` flag for 3D volumes

## Resource Requirements

- **Download**: ~4-8 hours, 16GB RAM, 4 CPUs
- **Processing**: ~2-6 hours, 32GB RAM, 8 CPUs
- **Training**: ~12-48 hours, 32GB RAM, 8 CPUs, 1 GPU

Adjust time limits in SLURM scripts based on your dataset size.

## Additional Resources

- [Sockeye Documentation](https://arc.ubc.ca/compute-storage/ubc-arc-sockeye)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
- [TCIA Website](https://www.cancerimagingarchive.net/)
- [NBIA Data Retriever Guide](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)

## Notes

- Use `$SCRATCH` for large datasets (downloaded and processed data)
- Use `$HOME` for project code and outputs
- Check disk quotas: `quota -s`
- Monitor job resources: `sstat -j <job_id>`


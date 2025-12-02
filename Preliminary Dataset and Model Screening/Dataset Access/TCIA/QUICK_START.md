# Quick Start Guide: TCIA Training on Sockeye

## Overview

This guide provides a quick reference for setting up and running training on UBC Sockeye using ISPY2 and DUKE datasets from TCIA.

## Prerequisites Checklist

- [ ] Sockeye account with GPU allocation
- [ ] NBIA Data Retriever downloaded (if using CLI method)
- [ ] Project code uploaded to Sockeye
- [ ] TCIA account (OPTIONAL - not required as of July 2025, only for legacy collections)

## Quick Setup (5 minutes)

```bash
# 1. Navigate to sockeye directory
cd "Preliminary Dataset and Model Screening/Dataset Access/TCIA/sockeye"

# 2. Run setup script
bash setup_environment.sh

# 3. Set TCIA credentials (OPTIONAL - not required as of July 2025)
# Only needed for legacy controlled-access collections
# export TCIA_USERNAME="your_username"
# export TCIA_PASSWORD="your_password"

# 4. Edit SLURM scripts to set your account
# Replace def-<your_account> with your actual account in all .slurm files
```

## Recommended: Test First!

Before downloading the full dataset, test with a small subset:

```bash
# Test download (5 patients each, ~30 min)
sbatch download_test_subset.slurm
squeue -u $USER

# After download completes, test processing
sbatch process_test_data.slurm
squeue -u $USER
```

If test succeeds, proceed with full dataset.

## Three-Step Workflow (Full Dataset)

### Step 1: Download (6-12 hours)

```bash
sbatch download_tcia_data.slurm
squeue -u $USER  # Monitor
```

### Step 2: Process (2-6 hours)

```bash
# Wait for download to complete, then:
sbatch process_tcia_data.slurm
squeue -u $USER  # Monitor
```

### Step 3: Train (12-48 hours)

```bash
# Wait for processing to complete, then:
sbatch train_resnet.slurm
squeue -u $USER  # Monitor
```

## File Locations

- **Downloaded data**: `$SCRATCH/tcia_data/`
- **Processed data**: `$SCRATCH/tcia_processed/`
- **Training outputs**: `$SCRATCH/resnet_outputs/`
- **Logs**: `sockeye/logs/`

## Common Commands

```bash
# Check job status
squeue -u $USER

# Cancel a job
scancel <job_id>

# View logs
tail -f logs/download_*.log
tail -f logs/process_*.log
tail -f logs/train_*.log

# Check disk usage
quota -s
du -sh $SCRATCH/tcia_data
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Job pending | Check queue: `squeue -u $USER` |
| Out of memory | Increase `--mem` in SLURM script |
| GPU not found | Verify `--gres=gpu:1` in training script |
| NBIA not found | Download and add to PATH or specify path |
| DICOM errors | Check metadata CSV format |

## Important Notes

1. **Storage**: Use `$SCRATCH` for large datasets (not `$HOME`)
2. **Credentials**: Never commit TCIA credentials to git
3. **Time limits**: Adjust `--time` in SLURM scripts based on dataset size
4. **Metadata**: You may need to create CSV files with patient labels

## Next Steps

After training completes:
1. Check outputs in `$SCRATCH/resnet_outputs/`
2. Review training logs for metrics
3. Download model checkpoints if needed
4. Evaluate on test set

## Getting Help

- Sockeye docs: https://arc.ubc.ca/compute-storage/ubc-arc-sockeye
- TCIA docs: https://www.cancerimagingarchive.net/
- Check logs in `sockeye/logs/` directory


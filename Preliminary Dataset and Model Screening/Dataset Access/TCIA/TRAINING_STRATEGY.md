# Training Strategy: Full Dataset vs Incremental

## Recommendation: Download Entire Dataset First

For final training with ISPY2 and DUKE datasets, **download the entire dataset first**, then train. This is the standard and recommended approach.

## Why Download Full Dataset First?

### 1. **Proper Train/Val/Test Splits**
- Your code uses `sklearn.model_selection.train_test_split()` which requires the full dataset
- Ensures balanced, stratified splits across classes
- Prevents data leakage (same patient in train and val)
- Critical for medical imaging where patient-level splits matter

### 2. **Reproducibility**
- Fixed random seed ensures same splits every time
- Essential for comparing different models/hyperparameters
- Required for publication and peer review

### 3. **Data Quality & Analysis**
- Can analyze full dataset statistics before training
- Detect class imbalances, missing data, outliers
- Plan preprocessing and augmentation strategies

### 4. **Standard Deep Learning Practice**
- PyTorch DataLoader expects full dataset structure
- Better shuffling across entire dataset each epoch
- Proper early stopping based on validation set
- Standard approach in all major ML frameworks

### 5. **Better Model Performance**
- Model sees all data patterns during training
- Better generalization from diverse samples
- More stable training dynamics

## Workflow for Full Dataset Training

### Step 1: Download Entire Dataset
```bash
# On Sockeye - this may take several hours
sbatch download_tcia_data.slurm

# Monitor progress
squeue -u $USER
tail -f logs/download_*.log
```

**Expected sizes:**
- ISPY2: ~50-200 GB (depending on series)
- DUKE: ~20-100 GB
- Combined: ~70-300 GB

### Step 2: Process All Data
```bash
# Convert DICOM to NIfTI and create train/val splits
sbatch process_tcia_data.slurm
```

This creates proper train/val splits from the full dataset.

### Step 3: Train on Full Dataset
```bash
# Train with entire processed dataset
sbatch train_resnet.slurm
```

## Storage Considerations on Sockeye

### Use `$SCRATCH` for Large Datasets
- `$SCRATCH` has much larger quotas than `$HOME`
- Designed for large temporary datasets
- Faster I/O for training

```bash
# In SLURM scripts, use:
export DATA_DIR="${SCRATCH}/tcia_data"        # Downloads
export PROCESSED_DIR="${SCRATCH}/tcia_processed"  # Processed
export OUTPUT_DIR="${SCRATCH}/resnet_outputs"    # Training outputs
```

### Check Your Quotas
```bash
# Check available space
quota -s
df -h $SCRATCH
```

### If Storage is Limited
If you hit storage limits, consider:
1. **Process and delete raw DICOM** after conversion to NIfTI
2. **Extract center slices only** (saves ~90% space vs full 3D volumes)
3. **Download one dataset at a time** (ISPY2, then DUKE)
4. **Use compression** for processed data

## When Incremental Training Might Be Considered

### Scenario: Extremely Large Datasets (500+ GB)
If storage is truly limited, you could:
1. Download in batches
2. Process each batch
3. Train incrementally

**However, this has significant drawbacks:**
- ❌ Can't do proper train/val split
- ❌ Harder to reproduce results
- ❌ May not converge as well
- ❌ More complex implementation
- ❌ Can't analyze full dataset

### Hybrid Approach (If Needed)
If you must use incremental training:

```python
# Pseudo-code for incremental training
for batch_num in range(num_batches):
    # Download batch
    download_batch(batch_num)
    
    # Process batch
    process_batch(batch_num)
    
    # Train on batch (accumulate gradients)
    train_on_batch(batch_num)
    
    # Delete raw data to save space
    cleanup_batch(batch_num)
```

**But this is NOT recommended** for your use case.

## Recommended Approach for Your Project

### For Initial Screening/Prototyping
- Download small subset (5-10 patients per class)
- Quick iteration and testing
- Use your existing `download_malignant_samples.py` approach

### For Final Training
- Download **entire ISPY2 and DUKE datasets**
- Process all data at once
- Create proper train/val/test splits
- Train on full dataset

## Time Estimates

### Download Times (on Sockeye)
- ISPY2: 4-8 hours
- DUKE: 2-4 hours
- **Total: 6-12 hours** (can run overnight)

### Processing Times
- DICOM to NIfTI conversion: 2-6 hours
- Creating splits: < 1 hour
- **Total: 2-7 hours**

### Training Times
- ResNet50 on full dataset: 12-48 hours (depending on size)
- With early stopping: Often completes faster

## Best Practices

1. **Download once, train many times**
   - Download and process data once
   - Reuse processed data for multiple training runs
   - Experiment with hyperparameters without re-downloading

2. **Keep processed data, delete raw DICOM**
   - NIfTI files are smaller and faster to load
   - Keep processed data for future experiments
   - Delete raw DICOM to save space

3. **Use proper train/val/test splits**
   - Train: 80% of patients
   - Val: 10% of patients (for early stopping)
   - Test: 10% of patients (final evaluation only)

4. **Monitor storage**
   - Check quotas regularly
   - Clean up old outputs
   - Archive important results

## Summary

✅ **Download entire dataset first** - Standard, recommended approach
✅ **Process all data** - Create proper splits
✅ **Train on full dataset** - Best performance and reproducibility

❌ **Avoid incremental training** - Unless absolutely necessary due to storage constraints

For ISPY2 + DUKE (~70-300 GB), this is manageable on Sockeye's `$SCRATCH` storage.


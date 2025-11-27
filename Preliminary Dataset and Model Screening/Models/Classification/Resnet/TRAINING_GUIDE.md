# ResNet Training Guide - Transfer Learning with Odelia Data

## Quick Start: Transfer Learning (Recommended)

### Option 1: Use the Helper Script (Easiest)

```bash
cd "Preliminary Dataset and Model Screening/Models/Classification/Resnet"
python train_odelia_transfer_learning.py
```

This script automatically:
- ✅ Loads settings from `config_odelia.json`
- ✅ Enables transfer learning (pretrained ImageNet weights)
- ✅ Uses the Odelia data directory
- ✅ Applies all recommended settings

### Option 2: Command Line (Full Control)

```bash
cd "Preliminary Dataset and Model Screening/Models/Classification/Resnet"

python train_resnet50.py \
    --data-dir "../../Dataset Access/Odelia/resnet_data" \
    --input-channels 1 \
    --image-size 224 \
    --pretrained \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --scheduler plateau \
    --num-workers 4 \
    --output-dir "./outputs/breast_mri_resnet50_odelia" \
    --save-interval 10 \
    --early-stopping-patience 15 \
    --class-weights
```

## What is Transfer Learning?

**Transfer Learning** (`--pretrained`):
- ✅ Starts with ResNet50 weights pretrained on ImageNet (1.2M images, 1000 classes)
- ✅ Only the final classification layer is randomly initialized
- ✅ All other layers start with learned features from natural images
- ✅ **Faster convergence** - usually needs fewer epochs
- ✅ **Better performance** - especially with limited data
- ✅ **Recommended** for medical imaging tasks

**Training from Scratch** (`--no-pretrained`):
- ❌ All weights randomly initialized
- ❌ Requires more data and training time
- ❌ May not converge as well with small datasets

## How Transfer Learning Works in This Code

1. **Load Pretrained ResNet50**: `models.resnet50(pretrained=True)`
2. **Adapt First Layer**: Since MRI is grayscale (1 channel) vs ImageNet RGB (3 channels):
   - Replaces first conv layer to accept 1 channel
   - Initializes by averaging the 3 pretrained RGB channels
3. **Replace Final Layer**: Changes from 1000 ImageNet classes to 2 classes (benign/malignant)
4. **Fine-tune**: All layers are trainable and will adapt to your breast MRI data

## Important Notes

### Current Data Limitation
⚠️ **You only have 3 patients (15 images) in the train split and 0 in val split.**

For proper training, you should:
1. **Download more data** - modify `download_one_split_3_patients.py` to get more patients
2. **Create a validation split** - manually split your data or download from val/test splits
3. **Use data augmentation** - already enabled in the training script

### Recommended Next Steps

1. **Download more patients**:
   ```python
   # In download_one_split_3_patients.py, change:
   max_patients = 50  # or more
   ```

2. **Download validation split**:
   ```python
   # Change target_split to "val" and download some validation data
   target_split = "val"
   max_patients = 10
   ```

3. **Reorganize data**:
   ```bash
   python organize_for_resnet.py
   ```

## Training Output

After training, you'll find in `outputs/breast_mri_resnet50_odelia/`:
- `best_model.pt` - Model with lowest validation loss
- `best_f1_model.pt` - Model with highest F1 score
- `final_model.pt` - Final model after all epochs
- `training_history.png` - Training curves
- `training_history.json` - Detailed metrics
- `config.json` - Training configuration used

## Monitoring Training

The script will print:
- Training/validation loss, accuracy, precision, recall, F1 score
- Learning rate adjustments
- Best model saves
- Early stopping (if enabled)

## Troubleshooting

### Issue: "No class directories found in val/"
**Solution**: You need validation data. Either:
- Download validation split from Odelia
- Manually split your training data
- Temporarily use train data for validation (not recommended for production)

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size:
```bash
--batch-size 8  # or even 4
```

### Issue: Model not learning
**Solution**: 
- Check if you have enough data (need more than 3 patients)
- Verify labels are correct
- Try different learning rates (0.001, 0.0001, 0.00001)

## Advanced: Two-Stage Fine-Tuning

For even better results, you can do two-stage fine-tuning:

**Stage 1: Freeze backbone, train only classifier**
```python
# In train_resnet50.py, add after model creation:
for param in model.resnet.parameters():
    param.requires_grad = False
# Only train the final fc layer
```

**Stage 2: Unfreeze and fine-tune all layers**
```python
# Unfreeze all layers
for param in model.resnet.parameters():
    param.requires_grad = True
# Train with lower learning rate
```

This approach can give better results but requires more training time.


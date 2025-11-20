# Breast MRI Lesion Detection with ResNet50

This repository contains a complete training pipeline for breast MRI lesion detection using ResNet50.

*NOTE* reconfigure to match odelia streaming 

## Features

- **ResNet50 Architecture**: Pre-trained on ImageNet, fine-tuned for breast MRI lesion classification
- **Flexible Input**: Supports both grayscale (1-channel) and RGB (3-channel) MRI images
- **Data Augmentation**: Random flips, rotations, and color jittering
- **Class Balancing**: Optional class weights for imbalanced datasets
- **Comprehensive Metrics**: Tracks accuracy, precision, recall, and F1 score
- **Visualization**: Automatic generation of training history plots
- **Early Stopping**: Prevents overfitting with configurable patience
- **Model Checkpointing**: Saves best models and periodic checkpoints

## Dataset Structure

Organize your dataset in the following structure:

```
data/
    breast_mri/
        train/
            benign/          # or class0/
                image1.png
                image2.png
                ...
            malignant/       # or class1/
                image1.png
                image2.png
                ...
        val/
            benign/
                image1.png
                ...
            malignant/
                image1.png
                ...
```

### Supported Image Formats

- PNG, JPG, JPEG, TIF, TIFF (standard image formats)
- NIfTI (.nii, .nii.gz) - MRI volumes (middle slice will be extracted)

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy pillow tqdm scikit-learn matplotlib
pip install nibabel  # Required only if using NIfTI files
```

Or install from the existing requirements:

```bash
pip install -r ../AI-for-Sarcopenia-main/UI/requirements.txt
```

## Usage

### Basic Training

```bash
python train_resnet50.py --data-dir ./data/breast_mri --output-dir ./outputs
```

### Advanced Training with Custom Parameters

```bash
python train_resnet50.py \
    --data-dir ./data/breast_mri \
    --input-channels 1 \
    --image-size 224 \
    --pretrained \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --scheduler plateau \
    --early-stopping-patience 20 \
    --class-weights \
    --output-dir ./outputs/breast_mri_resnet50
```

### Using Configuration File

You can modify `config.json` and load it programmatically, or use it as a reference for command-line arguments.

## Command-Line Arguments

### Data Arguments
- `--data-dir`: Path to dataset directory (required)
- `--input-channels`: Number of input channels (1 for grayscale, 3 for RGB, default: 1)
- `--image-size`: Input image size (default: 224)

### Model Arguments
- `--pretrained`: Use pretrained ImageNet weights (default: True)
- `--no-pretrained`: Do not use pretrained weights

### Training Arguments
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay for regularization (default: 1e-4)
- `--scheduler`: Learning rate scheduler - 'plateau', 'step', or 'none' (default: 'plateau')
- `--step-size`: Step size for StepLR scheduler (default: 10)
- `--gamma`: Gamma for StepLR scheduler (default: 0.1)

### Other Arguments
- `--num-workers`: Number of data loading workers (default: 4)
- `--output-dir`: Output directory for models and logs (default: ./outputs)
- `--save-interval`: Save checkpoint every N epochs (default: 10)
- `--early-stopping-patience`: Early stopping patience (default: 15, 0 to disable)
- `--class-weights`: Use class weights to handle imbalanced dataset

## Output Files

After training, the following files will be saved in the output directory:

- `best_model.pt`: Model with lowest validation loss
- `best_f1_model.pt`: Model with highest validation F1 score
- `final_model.pt`: Model from the last epoch
- `checkpoint_epoch_N.pt`: Periodic checkpoints
- `training_history.png`: Visualization of training metrics
- `training_history.json`: Training metrics in JSON format
- `config.json`: Training configuration

## Model Architecture

The model uses ResNet50 as the backbone:

1. **Input Layer**: Modified to accept 1-channel (grayscale) or 3-channel (RGB) images
2. **Backbone**: ResNet50 with ImageNet pretrained weights
3. **Classifier**: Fully connected layer with number of outputs equal to number of classes

For grayscale input, the first convolutional layer is modified to accept single-channel input, and weights are initialized by averaging the pretrained RGB channel weights.

## Training Tips

1. **Data Augmentation**: The default augmentation includes random flips, rotations, and color jittering. Adjust based on your data characteristics.

2. **Class Imbalance**: Use `--class-weights` flag if your dataset is imbalanced. The weights are automatically calculated from the training set.

3. **Learning Rate**: Start with 1e-4 for fine-tuning pretrained models. Adjust based on training dynamics.

4. **Batch Size**: Adjust based on your GPU memory. Larger batch sizes generally lead to more stable training.

5. **Early Stopping**: Monitor validation metrics and adjust patience based on your dataset size and training time.

## Evaluation

To evaluate a trained model, you can load the checkpoint:

```python
import torch
from train_resnet50 import ResNet50Classifier

# Load model
checkpoint = torch.load('outputs/best_model.pt')
model = ResNet50Classifier(num_classes=2, pretrained=False, input_channels=1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use model for inference
# ...
```

## Citation

If you use this code, please cite:

```bibtex
@article{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2016}
}
```

## License

This code is provided for research and educational purposes.


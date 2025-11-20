"""
Dataset loader for breast MRI lesion images
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from collections import Counter


class BreastMRIDataset(Dataset):
    """
    Dataset for breast MRI lesion images
    
    Expected directory structure:
    data_dir/
        train/
            class0/  (e.g., benign/)
                image1.png
                image2.png
                ...
            class1/  (e.g., malignant/)
                image1.png
                image2.png
                ...
        val/
            class0/
                image1.png
                ...
            class1/
                image1.png
                ...
    
    Alternative structure (if using a single directory with labels):
    data_dir/
        images/
            image1.png
            image2.png
            ...
        labels.csv  (with columns: filename, label)
    """
    
    def __init__(self, data_dir, split='train', transform=None, input_channels=1):
        """
        Args:
            data_dir: Path to dataset directory
            split: 'train' or 'val'
            transform: Optional transform to be applied on a sample
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.input_channels = input_channels
        
        # Load data
        self.images, self.labels, self.class_names = self._load_data()
        self.num_classes = len(self.class_names)
        
        print(f"Loaded {len(self.images)} images for {split} split")
        print(f"Classes: {self.class_names}")
    
    def _load_data(self):
        """Load images and labels from directory structure"""
        images = []
        labels = []
        
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Get class directories
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {split_dir}")
        
        class_names = [d.name for d in class_dirs]
        
        # Load images from each class directory
        for class_idx, class_dir in enumerate(class_dirs):
            # Supported image extensions
            extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.nii', '.nii.gz']
            
            image_files = []
            for ext in extensions:
                image_files.extend(list(class_dir.glob(f'*{ext}')))
                image_files.extend(list(class_dir.glob(f'*{ext.upper()}')))
            
            for img_path in image_files:
                images.append(str(img_path))
                labels.append(class_idx)
        
        return images, labels, class_names
    
    def _load_image(self, img_path):
        """Load and preprocess image"""
        img_path = Path(img_path)
        
        # Handle NIfTI files (MRI format)
        if img_path.suffix in ['.nii', '.gz']:
            try:
                import nibabel as nib
                nii_img = nib.load(str(img_path))
                img_array = nii_img.get_fdata()
                
                # Handle 3D volumes - take middle slice
                if len(img_array.shape) == 3:
                    slice_idx = img_array.shape[2] // 2
                    img_array = img_array[:, :, slice_idx]
                
                # Normalize to 0-255
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
                img_array = (img_array * 255).astype(np.uint8)
                
                # Convert to PIL Image
                img = Image.fromarray(img_array, mode='L')
            except ImportError:
                raise ImportError("nibabel is required for NIfTI files. Install with: pip install nibabel")
            except Exception as e:
                raise ValueError(f"Error loading NIfTI file {img_path}: {e}")
        else:
            # Load regular image file
            img = Image.open(img_path)
            
            # Convert to grayscale if needed
            if self.input_channels == 1 and img.mode != 'L':
                img = img.convert('L')
            elif self.input_channels == 3 and img.mode != 'RGB':
                img = img.convert('RGB')
        
        return img
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = self._load_image(img_path)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_class_counts(self):
        """Get count of samples per class"""
        return Counter(self.labels)
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets"""
        class_counts = self.get_class_counts()
        total = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = {}
        for class_idx, count in class_counts.items():
            weights[class_idx] = total / (num_classes * count)
        
        return weights


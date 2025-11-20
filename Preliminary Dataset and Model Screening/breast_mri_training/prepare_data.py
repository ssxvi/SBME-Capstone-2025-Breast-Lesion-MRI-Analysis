"""
Helper script to organize breast MRI dataset into train/val structure
"""

import os
import shutil
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
import random


def organize_data(source_dir, output_dir, train_ratio=0.8, random_seed=42):
    """
    Organize dataset into train/val structure
    
    Expected source structure:
    source_dir/
        class0/
            image1.png
            image2.png
            ...
        class1/
            image1.png
            ...
    
    Output structure:
    output_dir/
        train/
            class0/
            class1/
        val/
            class0/
            class1/
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Get class directories
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    class_dirs = sorted(class_dirs)
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {source_dir}")
    
    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Get all image files
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.nii', '.nii.gz']
        image_files = []
        for ext in extensions:
            image_files.extend(list(class_dir.glob(f'*{ext}')))
            image_files.extend(list(class_dir.glob(f'*{ext.upper()}')))
        
        image_files = sorted(image_files)
        
        if len(image_files) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        print(f"\nProcessing {class_name}: {len(image_files)} images")
        
        # Split into train and val
        random.seed(random_seed)
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * train_ratio)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        # Create class directories
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        
        # Copy train images
        print(f"  Train: {len(train_images)} images")
        for img_path in train_images:
            shutil.copy2(img_path, train_class_dir / img_path.name)
        
        # Copy val images
        print(f"  Val: {len(val_images)} images")
        for img_path in val_images:
            shutil.copy2(img_path, val_class_dir / img_path.name)
    
    print(f"\n✓ Dataset organized successfully!")
    print(f"  Train directory: {train_dir}")
    print(f"  Val directory: {val_dir}")


def main():
    parser = argparse.ArgumentParser(description='Organize breast MRI dataset into train/val structure')
    parser.add_argument('--source-dir', type=str, required=True,
                        help='Source directory with class subdirectories')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for organized dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Ratio of data for training (default: 0.8)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for splitting (default: 42)')
    
    args = parser.parse_args()
    
    organize_data(
        args.source_dir,
        args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()


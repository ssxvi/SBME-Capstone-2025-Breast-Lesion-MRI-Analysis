"""
Quick script to create a validation split from training data
Moves 20% of training data to validation split
"""

from pathlib import Path
import shutil
import random

def create_val_split(data_dir="./resnet_data", val_ratio=0.2):
    """
    Create validation split by moving some files from train to val
    
    Args:
        data_dir: Root directory with train/ and val/ subdirectories
        val_ratio: Ratio of data to move to validation (default: 0.2 = 20%)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    # Ensure val directories exist
    for class_name in ['class0', 'class1']:
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for class_name in ['class0', 'class1']:
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        
        if not train_class_dir.exists():
            continue
        
        # Get all files
        all_files = list(train_class_dir.glob("*.nii.gz"))
        
        if len(all_files) == 0:
            print(f"No files found in {train_class_dir}")
            continue
        
        # Calculate number to move
        num_to_move = max(1, int(len(all_files) * val_ratio))
        
        # Randomly select files to move
        random.seed(42)  # For reproducibility
        files_to_move = random.sample(all_files, min(num_to_move, len(all_files)))
        
        # Move files
        moved = 0
        for file_path in files_to_move:
            target_path = val_class_dir / file_path.name
            shutil.move(str(file_path), str(target_path))
            moved += 1
        
        print(f"{class_name}: Moved {moved}/{len(all_files)} files to validation")
    
    print(f"\nValidation split created in {val_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create validation split from training data')
    parser.add_argument('--data-dir', type=str, default='./resnet_data',
                        help='Root directory with train/val subdirectories')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of data to move to validation (default: 0.2)')
    
    args = parser.parse_args()
    create_val_split(args.data_dir, args.val_ratio)


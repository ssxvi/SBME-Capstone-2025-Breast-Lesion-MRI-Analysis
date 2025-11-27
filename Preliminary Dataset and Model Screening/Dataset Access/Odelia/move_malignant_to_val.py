"""
Move some malignant training samples to validation split
"""

from pathlib import Path
import shutil
import random

def move_malignant_to_val(
    data_dir="C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\Preliminary Dataset and Model Screening\\Dataset Access\\Odelia\\resnet_data",
    val_ratio=0.2,
    class_name="class1"
):
    """
    Move some malignant training samples to validation
    
    Args:
        data_dir: Root directory with train/ and val/ subdirectories
        val_ratio: Ratio of data to move to validation (default: 0.2 = 20%)
        class_name: Class to move (default: "class1" for malignant)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train" / class_name
    val_dir = data_dir / "val" / class_name
    
    # Ensure val directory exists
    val_dir.mkdir(parents=True, exist_ok=True)
    
    if not train_dir.exists():
        print(f"Training directory not found: {train_dir}")
        return
    
    # Get all files
    all_files = list(train_dir.glob("*.nii.gz"))
    
    if len(all_files) == 0:
        print(f"No files found in {train_dir}")
        return
    
    # Calculate number to move
    num_to_move = max(1, int(len(all_files) * val_ratio))
    
    print(f"Found {len(all_files)} malignant images in training")
    print(f"Moving {num_to_move} images ({val_ratio*100:.1f}%) to validation")
    
    # Randomly select files to move
    random.seed(42)  # For reproducibility
    files_to_move = random.sample(all_files, min(num_to_move, len(all_files)))
    
    # Move files
    moved = 0
    for file_path in files_to_move:
        target_path = val_dir / file_path.name
        shutil.move(str(file_path), str(target_path))
        moved += 1
    
    print(f"\nSuccessfully moved {moved}/{len(all_files)} malignant images to validation")
    print(f"  Training: {len(all_files) - moved} images remaining")
    print(f"  Validation: {moved} images")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Move malignant training samples to validation')
    parser.add_argument('--data-dir', type=str, 
                        default='C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\Preliminary Dataset and Model Screening\\Dataset Access\\Odelia\\resnet_data',
                        help='Root directory with train/val subdirectories')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of data to move to validation (default: 0.2 = 20%%)')
    parser.add_argument('--class', type=str, default='class1', dest='class_name',
                        help='Class to move (default: class1 for malignant)')
    
    args = parser.parse_args()
    move_malignant_to_val(args.data_dir, args.val_ratio, args.class_name)


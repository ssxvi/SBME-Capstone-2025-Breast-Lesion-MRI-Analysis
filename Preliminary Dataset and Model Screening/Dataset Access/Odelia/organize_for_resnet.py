"""
Script to organize Odelia dataset into ResNet-compatible structure

Converts from:
    dataset_downloaded/INSTITUTION/data_unilateral/UID/*.nii.gz
    
To:
    resnet_data/
        train/
            class0/  (benign - Lesion=0)
            class1/  (malignant - Lesion=1)
        val/
            class0/
            class1/
"""

from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm


def organize_odelia_for_resnet(
    odelia_root: str = "C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\Preliminary Dataset and Model Screening\\Dataset Access\\Odelia\\dataset_downloaded",
    output_dir: str = "C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\Preliminary Dataset and Model Screening\\Dataset Access\\Odelia\\resnet_data",
    config: str = "unilateral"
):
    """
    Organize Odelia dataset for ResNet training
    
    Args:
        odelia_root: Root directory of downloaded Odelia data
        output_dir: Output directory for ResNet-compatible structure
        config: "unilateral" or "default"
    """
    odelia_root = Path(odelia_root)
    output_dir = Path(output_dir)
    
    dir_config = {
        "default": {
            "data": "data",
            "metadata": "metadata",
        },
        "unilateral": {
            "data": "data_unilateral",
            "metadata": "metadata_unilateral",
        },
    }
    
    data_dir_name = dir_config[config]["data"]
    metadata_dir_name = dir_config[config]["metadata"]
    
    # Create output directories
    for split in ['train', 'val']:
        for class_name in ['class0', 'class1']:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each institution
    total_files = 0
    for institution_dir in odelia_root.iterdir():
        if not institution_dir.is_dir():
            continue
            
        institution = institution_dir.name
        print(f"\nProcessing institution: {institution}")
        
        # Load metadata
        metadata_path = institution_dir / metadata_dir_name / "annotation.csv"
        split_path = institution_dir / metadata_dir_name / "split.csv"
        
        if not metadata_path.exists() or not split_path.exists():
            print(f"  Warning: Metadata files not found for {institution}, skipping...")
            continue
        
        # Load annotation and split information
        df_anno = pd.read_csv(metadata_path)
        df_split = pd.read_csv(split_path)
        
        # Merge to get UID, Split, and Lesion label
        df = pd.merge(df_anno, df_split, on='UID', how='inner')
        
        # Determine label column (Lesion, Malignancy, etc.)
        label_column = None
        for col in ['Lesion', 'Malignancy', 'malignancy', 'Label', 'label']:
            if col in df.columns:
                label_column = col
                break
        
        if label_column is None:
            print(f"  Warning: No label column found in {institution}, skipping...")
            continue
        
        print(f"  Using label column: {label_column}")
        print(f"  Label distribution:")
        print(df[label_column].value_counts().to_string())
        
        # Process each patient
        data_dir = institution_dir / data_dir_name
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Organizing {institution}"):
            uid = row['UID']
            split = row['Split'].lower()  # train or val
            label = int(row[label_column])  # 0 or 1
            
            # Source directory for this patient
            patient_dir = data_dir / uid
            
            if not patient_dir.exists():
                print(f"    Warning: Patient directory not found: {patient_dir}")
                continue
            
            # Target class directory
            class_name = f"class{label}"
            target_dir = output_dir / split / class_name
            
            # Copy all NIfTI files from this patient
            nii_files = list(patient_dir.glob("*.nii.gz"))
            
            for nii_file in nii_files:
                # Create unique filename: UID_image_type.nii.gz
                target_filename = f"{uid}_{nii_file.stem.replace('.nii', '')}.nii.gz"
                target_path = target_dir / target_filename
                
                # Copy file
                shutil.copy2(nii_file, target_path)
                total_files += 1
        
        print(f"  Processed {len(df)} patients from {institution}")
    
    print(f"\n{'='*60}")
    print(f"Organization complete!")
    print(f"Total files organized: {total_files}")
    print(f"Output directory: {output_dir}")
    print(f"\nDirectory structure:")
    for split in ['train', 'val']:
        for class_name in ['class0', 'class1']:
            class_dir = output_dir / split / class_name
            file_count = len(list(class_dir.glob("*.nii.gz")))
            print(f"  {split}/{class_name}: {file_count} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize Odelia data for ResNet training')
    parser.add_argument('--odelia-root', type=str, 
                        default='C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\Preliminary Dataset and Model Screening\\Dataset Access\\Odelia\\dataset_downloaded',
                        help='Root directory of downloaded Odelia data')
    parser.add_argument('--output-dir', type=str, 
                        default='C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\Preliminary Dataset and Model Screening\\Dataset Access\\Odelia\\resnet_data',
                        help='Output directory for ResNet-compatible structure')
    parser.add_argument('--config', type=str, default='unilateral', choices=['unilateral', 'default'],
                        help='Dataset configuration')
    
    args = parser.parse_args()
    
    organize_odelia_for_resnet(
        odelia_root=args.odelia_root,
        output_dir=args.output_dir,
        config=args.config
    )


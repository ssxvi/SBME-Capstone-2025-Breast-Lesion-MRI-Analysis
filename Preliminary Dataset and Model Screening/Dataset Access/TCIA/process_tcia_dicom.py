"""
Process TCIA DICOM data and organize for ResNet training

This script:
1. Converts DICOM series to NIfTI format (or extracts center slices as PNG)
2. Organizes data into train/val splits with class0 (benign) and class1 (malignant)
3. Handles metadata from TCIA collections (ISPY2, DUKE, etc.)

Expected TCIA download structure:
    downloaded_data/
        PatientID_1/
            Series_1/
                *.dcm files
            Series_2/
                *.dcm files
        PatientID_2/
            ...

Output structure:
    processed_data/
        train/
            class0/  (benign)
            class1/  (malignant)
        val/
            class0/
            class1/
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split
import pydicom
from pydicom.errors import InvalidDicomError
import nibabel as nib
from scipy.ndimage import zoom


def load_dicom_series(dicom_dir):
    """
    Load all DICOM files from a directory and sort them by slice location
    
    Returns:
        List of pydicom Dataset objects sorted by slice location
    """
    dicom_dir = Path(dicom_dir)
    dicom_files = []
    
    # Find all DICOM files
    for file_path in dicom_dir.rglob('*.dcm'):
        if file_path.is_file():
            try:
                ds = pydicom.dcmread(str(file_path))
                if hasattr(ds, 'pixel_array'):
                    dicom_files.append(ds)
            except (InvalidDicomError, Exception):
                continue
    
    if len(dicom_files) == 0:
        return None
    
    # Sort by slice location
    try:
        dicom_files.sort(key=lambda x: float(x.SliceLocation) if hasattr(x, 'SliceLocation') else 0)
    except (AttributeError, ValueError):
        try:
            dicom_files.sort(key=lambda x: int(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        except (AttributeError, ValueError):
            pass
    
    return dicom_files


def dicom_to_nifti(dicom_files, output_path):
    """
    Convert DICOM series to NIfTI format
    
    Args:
        dicom_files: List of pydicom Dataset objects
        output_path: Output NIfTI file path
    """
    if dicom_files is None or len(dicom_files) == 0:
        return False
    
    # Extract pixel arrays
    pixel_arrays = []
    for ds in dicom_files:
        pixel_arrays.append(ds.pixel_array.astype(np.float32))
    
    # Stack into 3D volume
    volume = np.stack(pixel_arrays, axis=-1)
    
    # Get spacing information
    spacing = [1.0, 1.0, 1.0]
    if hasattr(dicom_files[0], 'PixelSpacing'):
        spacing[0] = float(dicom_files[0].PixelSpacing[0])
        spacing[1] = float(dicom_files[0].PixelSpacing[1])
    if hasattr(dicom_files[0], 'SliceThickness'):
        spacing[2] = float(dicom_files[0].SliceThickness)
    
    # Create affine matrix (identity for now, can be improved with DICOM header info)
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    
    # Create NIfTI image
    nii_img = nib.Nifti1Image(volume, affine)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii_img, str(output_path))
    
    return True


def get_center_slice_as_nifti(dicom_files, output_path):
    """
    Extract center slice from DICOM series and save as NIfTI
    
    Args:
        dicom_files: List of pydicom Dataset objects
        output_path: Output NIfTI file path
    """
    if dicom_files is None or len(dicom_files) == 0:
        return False
    
    # Get center slice
    center_idx = len(dicom_files) // 2
    center_slice = dicom_files[center_idx]
    
    # Extract pixel array
    pixel_array = center_slice.pixel_array.astype(np.float32)
    
    # Add z dimension (single slice)
    volume = pixel_array[:, :, np.newaxis]
    
    # Create affine
    spacing = [1.0, 1.0, 1.0]
    if hasattr(center_slice, 'PixelSpacing'):
        spacing[0] = float(center_slice.PixelSpacing[0])
        spacing[1] = float(center_slice.PixelSpacing[1])
    
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = 1.0
    
    # Create NIfTI image
    nii_img = nib.Nifti1Image(volume, affine)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii_img, str(output_path))
    
    return True


def load_metadata(metadata_path):
    """
    Load metadata CSV file with patient labels
    
    Expected columns: PatientID, Label (or Malignancy, Lesion, etc.)
    """
    if not Path(metadata_path).exists():
        return None
    
    df = pd.read_csv(metadata_path)
    
    # Find label column
    label_col = None
    for col in ['Label', 'label', 'Malignancy', 'malignancy', 'Lesion', 'lesion', 'Class', 'class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print(f"Warning: No label column found in {metadata_path}")
        return None
    
    # Find patient ID column
    patient_col = None
    for col in ['PatientID', 'patient_id', 'Patient_ID', 'UID', 'uid']:
        if col in df.columns:
            patient_col = col
            break
    
    if patient_col is None:
        print(f"Warning: No patient ID column found in {metadata_path}")
        return None
    
    return df[[patient_col, label_col]].rename(columns={patient_col: 'PatientID', label_col: 'Label'})


def process_tcia_data(input_dir, output_dir, metadata_path=None, train_ratio=0.8, 
                      extract_center_slice=True, random_seed=42):
    """
    Process TCIA DICOM data and organize for training
    
    Args:
        input_dir: Directory containing TCIA downloaded DICOM data
        output_dir: Output directory for organized data
        metadata_path: Path to CSV file with patient labels (optional)
        train_ratio: Ratio of data for training (default: 0.8)
        extract_center_slice: If True, extract only center slice; if False, convert full 3D volume
        random_seed: Random seed for train/val split
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val']:
        for class_name in ['class0', 'class1']:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Load metadata if provided
    metadata = None
    if metadata_path:
        metadata = load_metadata(metadata_path)
    
    # Find all patient directories
    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Process each patient
    processed_patients = []
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        patient_id = patient_dir.name
        
        # Get label from metadata
        label = None
        if metadata is not None:
            patient_row = metadata[metadata['PatientID'] == patient_id]
            if len(patient_row) > 0:
                label = int(patient_row.iloc[0]['Label'])
            else:
                print(f"Warning: No metadata found for patient {patient_id}, skipping...")
                continue
        else:
            # If no metadata, try to infer from directory name or use default
            # This is a placeholder - you may need to adjust based on your data
            print(f"Warning: No metadata provided, cannot determine label for {patient_id}")
            continue
        
        # Find DICOM series in patient directory
        series_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
        
        if len(series_dirs) == 0:
            # Check if DICOM files are directly in patient directory
            dicom_files = load_dicom_series(patient_dir)
            if dicom_files:
                series_dirs = [patient_dir]
        
        # Process each series
        for series_dir in series_dirs:
            dicom_files = load_dicom_series(series_dir)
            
            if dicom_files is None or len(dicom_files) == 0:
                continue
            
            # Convert to NIfTI
            if extract_center_slice:
                # Extract center slice only
                nifti_filename = f"{patient_id}_{series_dir.name}_center.nii.gz"
            else:
                # Full 3D volume
                nifti_filename = f"{patient_id}_{series_dir.name}.nii.gz"
            
            # Determine split (will be finalized after processing all patients)
            # For now, we'll process and then split
            temp_output = output_dir / "temp" / f"class{label}" / nifti_filename
            temp_output.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert DICOM to NIfTI
            if extract_center_slice:
                success = get_center_slice_as_nifti(dicom_files, temp_output)
            else:
                success = dicom_to_nifti(dicom_files, temp_output)
            
            if success:
                processed_patients.append({
                    'patient_id': patient_id,
                    'series': series_dir.name,
                    'label': label,
                    'file': temp_output
                })
    
    # Split into train/val
    if len(processed_patients) == 0:
        print("No patients processed successfully!")
        return
    
    # Group by patient for splitting
    patients_by_id = {}
    for item in processed_patients:
        pid = item['patient_id']
        if pid not in patients_by_id:
            patients_by_id[pid] = []
        patients_by_id[pid].append(item)
    
    # Split patients (not individual images) to avoid data leakage
    patient_ids = list(patients_by_id.keys())
    train_patients, val_patients = train_test_split(
        patient_ids, 
        test_size=1-train_ratio, 
        random_state=random_seed,
        shuffle=True
    )
    
    # Move files to final locations
    train_count = {'class0': 0, 'class1': 0}
    val_count = {'class0': 0, 'class1': 0}
    
    for item in processed_patients:
        pid = item['patient_id']
        label = item['label']
        class_name = f"class{label}"
        
        if pid in train_patients:
            split = 'train'
            train_count[class_name] += 1
        else:
            split = 'val'
            val_count[class_name] += 1
        
        # Move file to final location
        final_path = output_dir / split / class_name / item['file'].name
        shutil.move(str(item['file']), str(final_path))
    
    # Clean up temp directory
    temp_dir = output_dir / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Total patients: {len(patient_ids)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients: {len(val_patients)}")
    print(f"\nTrain files:")
    print(f"  class0 (benign): {train_count['class0']}")
    print(f"  class1 (malignant): {train_count['class1']}")
    print(f"\nVal files:")
    print(f"  class0 (benign): {val_count['class0']}")
    print(f"  class1 (malignant): {val_count['class1']}")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Process TCIA DICOM data and organize for ResNet training'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing TCIA downloaded DICOM data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for organized data'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='Path to CSV file with patient labels (columns: PatientID, Label)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Ratio of data for training (default: 0.8)'
    )
    parser.add_argument(
        '--full-volume',
        action='store_true',
        help='Convert full 3D volumes (default: extract center slice only)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for train/val split (default: 42)'
    )
    
    args = parser.parse_args()
    
    process_tcia_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        train_ratio=args.train_ratio,
        extract_center_slice=not args.full_volume,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()


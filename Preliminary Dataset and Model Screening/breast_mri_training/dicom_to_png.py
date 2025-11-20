"""
Script to convert DICOM files to PNG images, extracting only the center slice
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import pydicom
from pydicom.errors import InvalidDicomError


def load_dicom_series(dicom_dir):
    """
    Load all DICOM files from a directory and sort them by slice location
    
    Args:
        dicom_dir: Directory containing DICOM files
    
    Returns:
        List of pydicom Dataset objects sorted by slice location
    """
    dicom_dir = Path(dicom_dir)
    dicom_files = []
    
    # Find all DICOM files
    for file_path in dicom_dir.rglob('*'):
        if file_path.is_file():
            try:
                # Try to read as DICOM
                ds = pydicom.dcmread(str(file_path))
                # Check if it has pixel data
                if hasattr(ds, 'pixel_array'):
                    dicom_files.append(ds)
            except (InvalidDicomError, Exception):
                # Skip non-DICOM files
                continue
    
    if len(dicom_files) == 0:
        raise ValueError(f"No valid DICOM files found in {dicom_dir}")
    
    # Sort by slice location if available, otherwise by instance number
    try:
        # Try to sort by SliceLocation (most reliable for multi-slice images)
        dicom_files.sort(key=lambda x: float(x.SliceLocation) if hasattr(x, 'SliceLocation') else 0)
    except (AttributeError, ValueError):
        try:
            # Fall back to InstanceNumber
            dicom_files.sort(key=lambda x: int(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        except (AttributeError, ValueError):
            # If neither is available, keep original order (might be single slice)
            pass
    
    return dicom_files


def get_center_slice(dicom_files):
    """
    Get the center slice from a list of DICOM files
    
    Args:
        dicom_files: List of pydicom Dataset objects
    
    Returns:
        pydicom Dataset object for the center slice
    """
    if len(dicom_files) == 0:
        raise ValueError("No DICOM files provided")
    
    center_idx = len(dicom_files) // 2
    return dicom_files[center_idx]


def dicom_to_array(dicom_slice):
    """
    Convert DICOM slice to numpy array with proper windowing
    
    Args:
        dicom_slice: pydicom Dataset object
    
    Returns:
        numpy array with values normalized to 0-255
    """
    # Get pixel array
    pixel_array = dicom_slice.pixel_array.astype(np.float32)
    
    # Apply windowing if available (WindowCenter and WindowWidth)
    if hasattr(dicom_slice, 'WindowCenter') and hasattr(dicom_slice, 'WindowWidth'):
        window_center = float(dicom_slice.WindowCenter[0] if isinstance(dicom_slice.WindowCenter, (list, tuple)) else dicom_slice.WindowCenter)
        window_width = float(dicom_slice.WindowWidth[0] if isinstance(dicom_slice.WindowWidth, (list, tuple)) else dicom_slice.WindowWidth)
        
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        # Apply windowing
        pixel_array = np.clip(pixel_array, window_min, window_max)
        pixel_array = (pixel_array - window_min) / (window_max - window_min + 1e-8)
    else:
        # No windowing information, normalize to min-max
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min + 1e-8)
        else:
            pixel_array = np.zeros_like(pixel_array)
    
    # Convert to 0-255 range
    pixel_array = (pixel_array * 255).astype(np.uint8)
    
    return pixel_array


def convert_dicom_to_png(input_path, output_path, recursive=False):
    """
    Convert DICOM file(s) to PNG, extracting center slice
    
    Args:
        input_path: Path to DICOM file or directory containing DICOM files
        output_path: Output PNG file path or directory
        recursive: If True, search recursively for DICOM files
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if input_path.is_file():
        # Single file mode
        try:
            dicom_slice = pydicom.dcmread(str(input_path))
            if not hasattr(dicom_slice, 'pixel_array'):
                print(f"Warning: {input_path} does not contain pixel data, skipping")
                return
            
            pixel_array = dicom_to_array(dicom_slice)
            img = Image.fromarray(pixel_array, mode='L')
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            print(f"Converted: {input_path.name} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    
    elif input_path.is_dir():
        # Directory mode
        if recursive:
            dicom_dirs = [d for d in input_path.rglob('*') if d.is_dir()]
            dicom_dirs.insert(0, input_path)  # Include root directory
        else:
            dicom_dirs = [input_path]
        
        for dicom_dir in dicom_dirs:
            try:
                # Load DICOM series
                dicom_files = load_dicom_series(dicom_dir)
                
                if len(dicom_files) == 0:
                    continue
                
                # Get center slice
                center_slice = get_center_slice(dicom_files)
                pixel_array = dicom_to_array(center_slice)
                img = Image.fromarray(pixel_array, mode='L')
                
                # Determine output path
                if output_path.is_dir():
                    # If output is a directory, preserve input structure
                    rel_path = dicom_dir.relative_to(input_path)
                    output_file = output_path / rel_path / f"{dicom_dir.name}_center_slice.png"
                else:
                    # Single output file (shouldn't happen in directory mode, but handle it)
                    output_file = output_path
                
                # Save
                output_file.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_file)
                print(f"Converted: {dicom_dir.name} (slice {len(dicom_files)//2 + 1}/{len(dicom_files)}) -> {output_file}")
                
            except Exception as e:
                print(f"Error processing {dicom_dir}: {e}")
    else:
        raise ValueError(f"Input path {input_path} does not exist")


def main():
    parser = argparse.ArgumentParser(
        description='Convert DICOM files to PNG images, extracting center slice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single DICOM file to PNG
  python dicom_to_png.py input.dcm output.png
  
  # Convert all DICOM files in a directory (single center slice per directory)
  python dicom_to_png.py input_dir/ output_dir/
  
  # Convert recursively
  python dicom_to_png.py input_dir/ output_dir/ --recursive
        """
    )
    
    parser.add_argument('input', type=str,
                        help='Input DICOM file or directory containing DICOM files')
    parser.add_argument('output', type=str,
                        help='Output PNG file or directory')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search recursively for DICOM files in subdirectories')
    
    args = parser.parse_args()
    
    convert_dicom_to_png(args.input, args.output, recursive=args.recursive)
    print("\n✓ Conversion complete!")


if __name__ == '__main__':
    main()


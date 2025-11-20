"""
Script to convert NRRD files to PNG images, extracting only the center slice
"""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def load_nrrd(file_path):
    """
    Load NRRD file using pynrrd or SimpleITK
    
    Args:
        file_path: Path to NRRD file
    
    Returns:
        numpy array of the image data
        header information dict (if available)
    """
    file_path = Path(file_path)
    
    # Try pynrrd first (native NRRD reader)
    try:
        import nrrd
        data, header = nrrd.read(str(file_path))
        return data, header
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Error loading with pynrrd: {e}")
        pass
    
    # Try SimpleITK as fallback
    try:
        import SimpleITK as sitk
        sitk_image = sitk.ReadImage(str(file_path))
        data = sitk.GetArrayFromImage(sitk_image)
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        
        header = {
            'spacing': spacing,
            'origin': origin,
            'direction': direction
        }
        
        return data, header
    except ImportError:
        raise ImportError(
            "Neither pynrrd nor SimpleITK is installed. "
            "Install one with: pip install pynrrd or pip install SimpleITK"
        )
    except Exception as e:
        raise ValueError(f"Error loading NRRD file: {e}")


def get_center_slice(data, view='axial'):
    """
    Extract center slice from 3D volume
    
    Args:
        data: 3D numpy array
        view: View plane ('axial', 'coronal', 'sagittal')
    
    Returns:
        2D numpy array of the center slice
    """
    if len(data.shape) == 2:
        # Already 2D
        return data
    elif len(data.shape) == 3:
        # 3D volume (Z, Y, X) - typical for medical images
        if view == 'axial':
            slice_idx = data.shape[0] // 2
            return data[slice_idx, :, :]
        elif view == 'coronal':
            slice_idx = data.shape[1] // 2
            return data[:, slice_idx, :]
        elif view == 'sagittal':
            slice_idx = data.shape[2] // 2
            return data[:, :, slice_idx]
    elif len(data.shape) == 4:
        # 4D volume (e.g., time series) - take first volume
        print("Warning: 4D volume detected. Taking first volume and center slice.")
        data = data[0]
        if view == 'axial':
            slice_idx = data.shape[0] // 2
            return data[slice_idx, :, :]
        elif view == 'coronal':
            slice_idx = data.shape[1] // 2
            return data[:, slice_idx, :]
        elif view == 'sagittal':
            slice_idx = data.shape[2] // 2
            return data[:, :, slice_idx]
    else:
        raise ValueError(f"Unsupported data dimensionality: {len(data.shape)}")


def apply_windowing(data, window_level=None, window_width=None):
    """
    Apply windowing to image data. If no windowing parameters provided,
    uses min-max normalization.
    
    Args:
        data: Input image array
        window_level: Window level (center). If None, uses min-max.
        window_width: Window width. If None, uses min-max.
    
    Returns:
        Windowed and normalized array (0-255 uint8)
    """
    if window_level is None or window_width is None:
        # Min-max normalization
        data_min = float(data.min())
        data_max = float(data.max())
        
        if data_max > data_min:
            normalized = ((data - data_min) / (data_max - data_min + 1e-8) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(data, dtype=np.uint8)
    else:
        # Apply windowing
        window_min = window_level - window_width / 2
        window_max = window_level + window_width / 2
        
        # Clip to window
        windowed = np.clip(data, window_min, window_max)
        
        # Normalize to 0-255
        if window_max > window_min:
            normalized = ((windowed - window_min) / (window_max - window_min + 1e-8) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(windowed, dtype=np.uint8)
    
    return normalized


def nrrd_to_array(nrrd_file, view='axial', window_level=None, window_width=None):
    """
    Convert NRRD file to numpy array (center slice, windowed)
    
    Args:
        nrrd_file: Path to NRRD file
        view: View plane ('axial', 'coronal', 'sagittal')
        window_level: Optional window level for windowing
        window_width: Optional window width for windowing
    
    Returns:
        numpy array with values normalized to 0-255
    """
    data, header = load_nrrd(nrrd_file)
    slice_data = get_center_slice(data, view=view)
    pixel_array = apply_windowing(slice_data, window_level, window_width)
    
    return pixel_array


def convert_nrrd_to_png(input_path, output_path, view='axial', recursive=False, 
                        window_level=None, window_width=None):
    """
    Convert NRRD file(s) to PNG, extracting center slice
    
    Args:
        input_path: Path to NRRD file or directory containing NRRD files
        output_path: Output PNG file path or directory
        view: View plane ('axial', 'coronal', 'sagittal')
        recursive: If True, search recursively for NRRD files
        window_level: Optional window level for windowing
        window_width: Optional window width for windowing
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if input_path.is_file():
        # Single file mode
        try:
            pixel_array = nrrd_to_array(input_path, view=view, 
                                       window_level=window_level, 
                                       window_width=window_width)
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
            nrrd_files = list(input_path.rglob('*.nrrd'))
            nrrd_files.extend(list(input_path.rglob('*.nhdr')))
        else:
            nrrd_files = list(input_path.glob('*.nrrd'))
            nrrd_files.extend(list(input_path.glob('*.nhdr')))
        
        if len(nrrd_files) == 0:
            print(f"No NRRD files found in {input_path}")
            return
        
        print(f"Found {len(nrrd_files)} NRRD files")
        
        for nrrd_file in nrrd_files:
            try:
                pixel_array = nrrd_to_array(nrrd_file, view=view,
                                           window_level=window_level,
                                           window_width=window_width)
                img = Image.fromarray(pixel_array, mode='L')
                
                # Determine output path
                if output_path.is_dir():
                    # If output is a directory, preserve input structure
                    rel_path = nrrd_file.relative_to(input_path)
                    output_file = output_path / rel_path.with_suffix('.png')
                    output_file = output_file.with_stem(output_file.stem + '_center_slice')
                else:
                    # Single output file (shouldn't happen in directory mode)
                    output_file = output_path
                
                # Save
                output_file.parent.mkdir(parents=True, exist_ok=True)
                img.save(output_file)
                print(f"Converted: {nrrd_file.name} -> {output_file}")
                
            except Exception as e:
                print(f"Error processing {nrrd_file}: {e}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NRRD files to PNG images, extracting center slice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single NRRD file to PNG
  python nrrd_to_png.py input.nrrd output.png
  
  # Convert with specific view
  python nrrd_to_png.py input.nrrd output.png --view coronal
  
  # Convert all NRRD files in a directory
  python nrrd_to_png.py input_dir/ output_dir/
  
  # Convert recursively with windowing
  python nrrd_to_png.py input_dir/ output_dir/ --recursive --window-level 50 --window-width 350
        """
    )
    
    parser.add_argument('input', type=str,
                        help='Input NRRD file or directory containing NRRD files')
    parser.add_argument('output', type=str,
                        help='Output PNG file or directory')
    parser.add_argument('--view', type=str, choices=['axial', 'coronal', 'sagittal'],
                        default='axial',
                        help='View plane for 3D volumes (default: axial)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search recursively for NRRD files in subdirectories')
    parser.add_argument('--window-level', type=float, default=None,
                        help='Window level for windowing (if not provided, uses min-max)')
    parser.add_argument('--window-width', type=float, default=None,
                        help='Window width for windowing (if not provided, uses min-max)')
    
    args = parser.parse_args()
    
    convert_nrrd_to_png(
        args.input, 
        args.output, 
        view=args.view,
        recursive=args.recursive,
        window_level=args.window_level,
        window_width=args.window_width
    )
    print("\n✓ Conversion complete!")


if __name__ == '__main__':
    main()


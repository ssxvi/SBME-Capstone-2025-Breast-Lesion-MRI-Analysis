"""
Interactive script to visualize NRRD files
Supports 3D volumes with slice navigation and windowing controls
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path
import sys


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
        print(f"Loaded NRRD file: {file_path.name}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        if 'space directions' in header:
            print(f"Spacing: {header.get('space directions', 'Unknown')}")
        return data, header
    except ImportError:
        pass
    except Exception as e:
        print(f"Error loading with pynrrd: {e}")
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
        
        print(f"Loaded NRRD file: {file_path.name}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Spacing: {spacing}")
        
        return data, header
    except ImportError:
        raise ImportError(
            "Neither pynrrd nor SimpleITK is installed. "
            "Install one with: pip install pynrrd or pip install SimpleITK"
        )
    except Exception as e:
        raise ValueError(f"Error loading NRRD file: {e}")


def apply_windowing(data, window_level, window_width):
    """
    Apply windowing to image data
    
    Args:
        data: Input image array
        window_level: Window level (center)
        window_width: Window width
    
    Returns:
        Windowed and normalized array (0-255 uint8)
    """
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2
    
    # Clip to window
    windowed = np.clip(data, window_min, window_max)
    
    # Normalize to 0-255
    if window_max > window_min:
        normalized = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(windowed, dtype=np.uint8)
    
    return normalized


class NRRDViewer:
    """Interactive viewer for NRRD files"""
    
    def __init__(self, file_path, view='axial'):
        """
        Initialize viewer
        
        Args:
            file_path: Path to NRRD file
            view: Initial view ('axial', 'coronal', 'sagittal')
        """
        self.file_path = Path(file_path)
        self.data, self.header = load_nrrd(file_path)
        
        # Determine initial windowing based on data range
        self.data_min = float(self.data.min())
        self.data_max = float(self.data.max())
        self.data_mean = float(self.data.mean())
        self.data_std = float(self.data.std())
        
        # Default windowing
        self.window_level = self.data_mean
        self.window_width = 2 * self.data_std if self.data_std > 0 else (self.data_max - self.data_min)
        
        # Handle different dimensionalities
        if len(self.data.shape) == 2:
            # 2D image
            self.view = '2d'
            self.num_slices = 1
        elif len(self.data.shape) == 3:
            # 3D volume (Z, Y, X) - typical for medical images
            self.view = view
            self.num_slices = {
                'axial': self.data.shape[0],
                'coronal': self.data.shape[1],
                'sagittal': self.data.shape[2]
            }[view]
            self.current_slice = self.num_slices // 2
        elif len(self.data.shape) == 4:
            # 4D volume (e.g., time series)
            print("Warning: 4D volume detected. Taking first volume.")
            self.data = self.data[0]
            self.view = view
            self.num_slices = {
                'axial': self.data.shape[0],
                'coronal': self.data.shape[1],
                'sagittal': self.data.shape[2]
            }[view]
            self.current_slice = self.num_slices // 2
        else:
            raise ValueError(f"Unsupported data dimensionality: {len(self.data.shape)}")
        
        # Setup figure
        self.setup_ui()
        self.update_display()
    
    def setup_ui(self):
        """Setup the matplotlib interface"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.subplots_adjust(bottom=0.25)
        
        # Image display
        self.im = self.ax.imshow(
            np.zeros((100, 100), dtype=np.uint8),
            cmap='gray',
            aspect='equal'
        )
        self.ax.set_title(f"{self.file_path.name}")
        self.ax.axis('off')
        
        # Slice slider (only for 3D)
        if self.view != '2d':
            ax_slice = plt.axes([0.2, 0.15, 0.6, 0.03])
            self.slider_slice = Slider(
                ax_slice,
                'Slice',
                0,
                self.num_slices - 1,
                valinit=self.current_slice,
                valfmt='%d'
            )
            self.slider_slice.on_changed(self.on_slice_change)
        
        # Window level slider
        ax_level = plt.axes([0.2, 0.10, 0.6, 0.03])
        self.slider_level = Slider(
            ax_level,
            'Window Level',
            self.data_min,
            self.data_max,
            valinit=self.window_level,
            valfmt='%.1f'
        )
        self.slider_level.on_changed(self.on_window_change)
        
        # Window width slider
        ax_width = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider_width = Slider(
            ax_width,
            'Window Width',
            1,
            self.data_max - self.data_min,
            valinit=self.window_width,
            valfmt='%.1f'
        )
        self.slider_width.on_changed(self.on_window_change)
        
        # Keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Instructions
        self.fig.text(0.5, 0.02, 
                     'Keyboard: ←/→ = previous/next slice, w/s = window level, a/d = window width, q = quit',
                     ha='center', fontsize=8)
    
    def get_current_slice_data(self):
        """Get the current slice based on view"""
        if self.view == '2d':
            return self.data
        elif self.view == 'axial':
            return self.data[self.current_slice, :, :]
        elif self.view == 'coronal':
            return self.data[:, self.current_slice, :]
        elif self.view == 'sagittal':
            return self.data[:, :, self.current_slice]
    
    def update_display(self):
        """Update the displayed image"""
        slice_data = self.get_current_slice_data()
        windowed = apply_windowing(slice_data, self.window_level, self.window_width)
        
        self.im.set_data(windowed)
        self.im.set_clim(0, 255)
        
        # Update title
        if self.view != '2d':
            view_name = self.view.capitalize()
            self.ax.set_title(
                f"{self.file_path.name} - {view_name} view, Slice {self.current_slice + 1}/{self.num_slices}"
            )
        else:
            self.ax.set_title(f"{self.file_path.name}")
        
        self.fig.canvas.draw_idle()
    
    def on_slice_change(self, val):
        """Handle slice slider change"""
        self.current_slice = int(self.slider_slice.val)
        self.update_display()
    
    def on_window_change(self, val):
        """Handle windowing slider changes"""
        self.window_level = self.slider_level.val
        self.window_width = self.slider_width.val
        self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard input"""
        if self.view == '2d':
            return
        
        if event.key == 'left' or event.key == 'right':
            # Change slice
            if event.key == 'left':
                self.current_slice = max(0, self.current_slice - 1)
            else:
                self.current_slice = min(self.num_slices - 1, self.current_slice + 1)
            
            self.slider_slice.set_val(self.current_slice)
        
        elif event.key == 'w':
            # Increase window level
            new_level = min(self.data_max, self.window_level + (self.data_max - self.data_min) * 0.05)
            self.slider_level.set_val(new_level)
        elif event.key == 's':
            # Decrease window level
            new_level = max(self.data_min, self.window_level - (self.data_max - self.data_min) * 0.05)
            self.slider_level.set_val(new_level)
        
        elif event.key == 'a':
            # Decrease window width
            new_width = max(1, self.window_width * 0.9)
            self.slider_width.set_val(new_width)
        elif event.key == 'd':
            # Increase window width
            new_width = min(self.data_max - self.data_min, self.window_width * 1.1)
            self.slider_width.set_val(new_width)
        
        elif event.key == 'q':
            # Quit
            plt.close('all')
            sys.exit(0)
    
    def show(self):
        """Show the viewer"""
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive viewer for NRRD files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View NRRD file (default axial view)
  python visualize_nrrd.py image.nrrd
  
  # View in coronal view
  python visualize_nrrd.py image.nrrd --view coronal
  
  # View in sagittal view
  python visualize_nrrd.py image.nrrd --view sagittal

Controls:
  - Sliders: Adjust slice, window level, and window width
  - Keyboard:
    ←/→ : Navigate between slices
    w/s : Increase/decrease window level
    a/d : Decrease/increase window width
    q   : Quit
        """
    )
    
    parser.add_argument('nrrd_file', type=str,
                        help='Path to NRRD file')
    parser.add_argument('--view', type=str, choices=['axial', 'coronal', 'sagittal'],
                        default='axial',
                        help='Initial viewing plane (default: axial)')
    
    args = parser.parse_args()
    
    if not Path(args.nrrd_file).exists():
        raise FileNotFoundError(f"NRRD file not found: {args.nrrd_file}")
    
    viewer = NRRDViewer(args.nrrd_file, view=args.view)
    viewer.show()


if __name__ == '__main__':
    main()


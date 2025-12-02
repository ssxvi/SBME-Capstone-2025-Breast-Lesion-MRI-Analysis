"""
Download ISPY2 breast MRI dataset from TCIA using NBIA Data Retriever

This script uses the NBIA Data Retriever CLI tool to download the ISPY2 dataset.
The NBIA tool must be installed.

Requirements:
- NBIA Data Retriever installed (download from TCIA website)
- Collection name: "ISPY2" (verify on TCIA website)

Note: As of July 2025, TCIA no longer requires account registration for most datasets.
Credentials are optional - only needed for legacy controlled-access collections.

Usage:
    # Without credentials (recommended for most datasets):
    python download-ispy2.py --output-dir /path/to/output
    
    # With credentials (only if needed for specific collections):
    python download-ispy2.py --username YOUR_TCIA_USERNAME --password YOUR_PASSWORD --output-dir /path/to/output
"""

import subprocess
import sys
import argparse
from pathlib import Path
import os
from tqdm import tqdm


def find_nbia_retriever():
    """
    Find NBIA Data Retriever executable
    Checks common installation locations
    """
    # Common locations for NBIA Data Retriever
    possible_paths = [
        "NBIA-Data-Retriever",
        "NBIA-Data-Retriever.exe",
        "/usr/local/bin/NBIA-Data-Retriever",
        os.path.expanduser("~/NBIA-Data-Retriever"),
        os.path.expanduser("~/NBIA-Data-Retriever/NBIA-Data-Retriever"),
    ]
    
    # Check if in PATH
    import shutil
    nbia_path = shutil.which("NBIA-Data-Retriever")
    if nbia_path:
        return nbia_path
    
    # Check common locations
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None


def download_collection(collection_name, output_dir, nbia_path=None, username=None, password=None):
    """
    Download a TCIA collection using NBIA Data Retriever
    
    Args:
        collection_name: Name of the TCIA collection (e.g., "ISPY2")
        output_dir: Output directory for downloaded data
        nbia_path: Path to NBIA Data Retriever executable (optional)
        username: TCIA username (optional, not required as of July 2025)
        password: TCIA password (optional, not required as of July 2025)
    """
    if nbia_path is None:
        nbia_path = find_nbia_retriever()
        if nbia_path is None:
            raise FileNotFoundError(
                "NBIA Data Retriever not found. Please install it from:\n"
                "https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images\n"
                "Or specify the path with --nbia-path"
            )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using NBIA Data Retriever: {nbia_path}")
    print(f"Collection: {collection_name}")
    print(f"Output directory: {output_dir}")
    
    # NBIA Data Retriever command format
    # Note: The exact command format may vary by version
    # As of July 2025, credentials are typically not required
    cmd = [
        nbia_path,
        "--collection", collection_name,
        "--outputPath", str(output_dir),
        "--downloadType", "series",  # or "patient" or "study"
    ]
    
    # Add credentials only if provided (for legacy controlled-access collections)
    if username and password:
        cmd.extend(["--username", username, "--password", password])
        print("\nExecuting command (with credentials):")
        print(" ".join(cmd[:4]) + " [USERNAME] [PASSWORD_HIDDEN] " + " ".join(cmd[6:]))
    else:
        print("\nExecuting command (no credentials required):")
        print(" ".join(cmd))
        print("Note: As of July 2025, TCIA no longer requires account registration for most datasets.")
    
    print("\nThis may take a while depending on dataset size...")
    
    try:
        # Run NBIA Data Retriever
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("\n✓ Download completed successfully!")
        print(f"Data saved to: {output_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n✗ Error during download:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        print("\nNote: Some versions of NBIA Data Retriever may require:")
        print("  1. Interactive login (run manually first)")
        print("  2. Different command-line arguments")
        print("  3. Configuration file instead of command-line args")
        return False
    except FileNotFoundError:
        print(f"\n✗ NBIA Data Retriever not found at: {nbia_path}")
        print("Please install NBIA Data Retriever or specify correct path")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download ISPY2 breast MRI dataset from TCIA using NBIA Data Retriever'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='ISPY2',
        help='TCIA collection name (default: ISPY2)'
    )
    parser.add_argument(
        '--username',
        type=str,
        default=None,
        help='TCIA username (optional - not required as of July 2025)'
    )
    parser.add_argument(
        '--password',
        type=str,
        default=None,
        help='TCIA password (optional - not required as of July 2025)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./ispy2_downloaded',
        help='Output directory for downloaded data (default: ./ispy2_downloaded)'
    )
    parser.add_argument(
        '--nbia-path',
        type=str,
        default=None,
        help='Path to NBIA Data Retriever executable (optional, will search if not provided)'
    )
    
    args = parser.parse_args()
    
    # Download collection
    success = download_collection(
        collection_name=args.collection,
        output_dir=args.output_dir,
        nbia_path=args.nbia_path,
        username=args.username,
        password=args.password
    )
    
    if success:
        print(f"\n{'='*60}")
        print("Next steps:")
        print(f"1. Verify downloaded data in: {args.output_dir}")
        print("2. Run data processing script to convert DICOM to training format:")
        print("   python process_tcia_dicom.py --input-dir <downloaded_dir> --output-dir <processed_dir>")
        print(f"{'='*60}")
    else:
        print("\nDownload failed. Please check:")
        print("1. Collection name is correct (check TCIA website)")
        print("2. NBIA Data Retriever is installed and accessible")
        if args.username and args.password:
            print("3. TCIA credentials are correct (if using legacy controlled-access collection)")
        sys.exit(1)


if __name__ == "__main__":
    main()


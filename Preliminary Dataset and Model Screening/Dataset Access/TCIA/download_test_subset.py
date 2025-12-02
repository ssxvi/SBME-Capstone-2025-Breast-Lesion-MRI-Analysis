"""
 ISPY2 and DUKE mini datasets download for testing workflow

This script downloads a limited number of patients/series to test the full pipeline
before downloading the entire dataset.

Usage:
    python download_test_subset.py --max-patients 5 --output-dir ./test_data
"""

import subprocess
import sys
import argparse
from pathlib import Path
import os
import time
import signal
from tqdm import tqdm


def find_nbia_retriever():
    """Find NBIA Data Retriever executable"""
    import shutil
    nbia_path = shutil.which("NBIA-Data-Retriever")
    if nbia_path:
        return nbia_path
    
    possible_paths = [
        "NBIA-Data-Retriever",
        "NBIA-Data-Retriever.exe",
        "/usr/local/bin/NBIA-Data-Retriever",
        os.path.expanduser("~/NBIA-Data-Retriever"),
        os.path.expanduser("~/NBIA-Data-Retriever/NBIA-Data-Retriever"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None


def count_patients_in_directory(data_dir):
    """Count number of patient directories in downloaded data"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return 0
    
    # Count top-level directories (typically patient directories)
    patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    return len(patient_dirs)


def download_with_limit(collection_name, output_dir, max_patients=5, max_time_minutes=30, 
                        nbia_path=None, username=None, password=None):
    """
    Download TCIA collection with limits on patients and time
    
    Args:
        collection_name: TCIA collection name
        output_dir: Output directory
        max_patients: Maximum number of patients to download
        max_time_minutes: Maximum time to allow for download
        nbia_path: Path to NBIA Data Retriever
        username: TCIA username (optional)
        password: TCIA password (optional)
    """
    if nbia_path is None:
        nbia_path = find_nbia_retriever()
        if nbia_path is None:
            raise FileNotFoundError(
                "NBIA Data Retriever not found. Please install it or specify path."
            )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"Downloading test subset: {collection_name}")
    print(f"Max patients: {max_patients}")
    print(f"Max time: {max_time_minutes} minutes")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # Build NBIA command
    cmd = [
        nbia_path,
        "--collection", collection_name,
        "--outputPath", str(output_dir),
        "--downloadType", "series",
    ]
    
    if username and password:
        cmd.extend(["--username", username, "--password", password])
        print("Using credentials for download")
    else:
        print("Downloading without credentials (not required as of July 2025)")
    
    print("\nStarting download...")
    print(f"Command: {' '.join(cmd[:4])} [args...]")
    print("\nMonitoring download progress...")
    print(f"Will stop after {max_patients} patients or {max_time_minutes} minutes\n")
    
    # Start download process
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Monitor progress
        last_count = 0
        check_interval = 10  # Check every 10 seconds
        
        while True:
            elapsed = time.time() - start_time
            
            # Check time limit
            if elapsed > max_time_seconds:
                print(f"\n⏱ Time limit reached ({max_time_minutes} minutes)")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                break
            
            # Check patient count
            current_count = count_patients_in_directory(output_dir)
            if current_count > last_count:
                print(f"  Downloaded {current_count} patients so far...")
                last_count = current_count
            
            if current_count >= max_patients:
                print(f"\n✓ Reached target of {max_patients} patients")
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                break
            
            # Check if process finished
            if process.poll() is not None:
                break
            
            time.sleep(check_interval)
        
        # Wait for process to finish
        _, _ = process.communicate(timeout=5)
        
        final_count = count_patients_in_directory(output_dir)
        elapsed_minutes = (time.time() - start_time) / 60
        
        print("\n" + "="*60)
        print("Download completed!")
        print(f"Patients downloaded: {final_count}")
        print(f"Time elapsed: {elapsed_minutes:.1f} minutes")
        print(f"Output directory: {output_dir}")
        print("="*60 + "\n")
        
        return final_count > 0
        
    except subprocess.TimeoutExpired:
        process.kill()
        final_count = count_patients_in_directory(output_dir)
        print(f"\nDownload stopped. Downloaded {final_count} patients.")
        return final_count > 0
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download small test subset of ISPY2 and DUKE datasets'
    )
    parser.add_argument(
        '--max-patients',
        type=int,
        default=5,
        help='Maximum number of patients to download per dataset (default: 5)'
    )
    parser.add_argument(
        '--max-time',
        type=int,
        default=30,
        help='Maximum download time in minutes per dataset (default: 30)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./tcia_test_data',
        help='Output directory for test data (default: ./tcia_test_data)'
    )
    parser.add_argument(
        '--username',
        type=str,
        default=None,
        help='TCIA username (optional)'
    )
    parser.add_argument(
        '--password',
        type=str,
        default=None,
        help='TCIA password (optional)'
    )
    parser.add_argument(
        '--nbia-path',
        type=str,
        default=None,
        help='Path to NBIA Data Retriever executable'
    )
    parser.add_argument(
        '--skip-ispy2',
        action='store_true',
        help='Skip ISPY2 download'
    )
    parser.add_argument(
        '--skip-duke',
        action='store_true',
        help='Skip DUKE download'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    ispy2_dir = output_dir / "ispy2"
    duke_dir = output_dir / "duke"
    
    separator = "="*60
    print(separator)
    print("TCIA Test Subset Download")
    print(separator)
    print(f"Max patients per dataset: {args.max_patients}")
    print(f"Max time per dataset: {args.max_time} minutes")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    results = {}
    
    # Download ISPY2
    if not args.skip_ispy2:
        print("\n" + "="*60)
        print("Downloading ISPY2 test subset...")
        print("="*60)
        success = download_with_limit(
            collection_name="ISPY2",
            output_dir=ispy2_dir,
            max_patients=args.max_patients,
            max_time_minutes=args.max_time,
            nbia_path=args.nbia_path,
            username=args.username,
            password=args.password
        )
        results['ISPY2'] = success
    else:
        print("\nSkipping ISPY2 download")
        results['ISPY2'] = None
    
    # Download DUKE
    if not args.skip_duke:
        print("\n" + "="*60)
        print("Downloading DUKE test subset...")
        print("="*60)
        success = download_with_limit(
            collection_name="Duke-Breast-Cancer-MRI",
            output_dir=duke_dir,
            max_patients=args.max_patients,
            max_time_minutes=args.max_time,
            nbia_path=args.nbia_path,
            username=args.username,
            password=args.password
        )
        results['DUKE'] = success
    else:
        print("\nSkipping DUKE download")
        results['DUKE'] = None
    
    # Summary
    separator = "="*60
    print("\n" + separator)
    print("Download Summary")
    print(separator)
    for dataset, success in results.items():
        if success is None:
            status = "Skipped"
        elif success:
            status = "✓ Success"
        else:
            status = "✗ Failed"
        print(f"{dataset:10s}: {status}")
    
    print(f"\nTest data location: {output_dir}")
    print("\nNext steps:")
    print("1. Verify downloaded data structure")
    print("2. Test processing script:")
    print(f"   python process_tcia_dicom.py --input-dir {ispy2_dir} --output-dir ./test_processed/ispy2")
    print(f"   python process_tcia_dicom.py --input-dir {duke_dir} --output-dir ./test_processed/duke")
    print("3. If successful, proceed with full dataset download")
    print(separator)


if __name__ == "__main__":
    main()


from pathlib import Path
from datasets import load_dataset
import torchio as tio 
import numpy as np 
import pandas as pd
from tqdm import tqdm
import sys
import subprocess


# --------------------- Settings ---------------------
repo_id = "ODELIA-AI/ODELIA-Challenge-2025"
config = "unilateral"  # "default" or "unilateral"
output_root = Path("C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\Preliminary Dataset and Model Screening\\Dataset Access\\Odelia\\dataset_downloaded")
target_split = "train"  # "train", "val", or "test"
max_patients = 3  # Number of patients to download
organize_for_resnet = True  # Automatically organize for ResNet after download
resnet_output_dir = Path("C:\\Programming\\SBME-Capstone-2025-Breast-Lesion-MRI-Analysis\\Preliminary Dataset and Model Screening\\Dataset Access\\Odelia\\resnet_data")  # Output directory for ResNet structure


# Load dataset in streaming mode
dataset = load_dataset(repo_id, name=config, streaming=True)

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

# Process dataset
metadata = []
num_patients = 0

# Only process the target split
if target_split not in dataset:
    available_splits = list(dataset.keys())
    raise ValueError(f"Split '{target_split}' not found. Available splits: {available_splits}")

split_dataset = dataset[target_split]
print(f"-------- Start Download - Split: {target_split} (max {max_patients} patients) --------")

for item in tqdm(split_dataset, desc="Downloading"):  # Stream data one-by-one
    if num_patients >= max_patients:
        print(f"\nReached limit of {max_patients} patients. Stopping download.")
        break
    
    uid = item["UID"]
    institution = item["Institution"]
    img_names = [name.split("Image_")[1] for name in item.keys() if name.startswith("Image")]
    
    # Create output folder
    path_folder = output_root / institution / dir_config[config]["data"] / uid
    path_folder.mkdir(parents=True, exist_ok=True)

    for img_name in img_names:
        img_data = item.pop(f"Image_{img_name}")
        img_affine = item.pop(f"Affine_{img_name}")

        # Skip if image data is None
        if img_data is None:
            continue

        # Extract image data and affine matrix
        img_data = np.array(img_data, dtype=np.int16) 
        img_affine = np.array(img_affine, dtype=np.float64)
        img = tio.ScalarImage(tensor=img_data, affine=img_affine)
        
        # Save image
        img.save(path_folder / f"{img_name}.nii.gz")

    # Store metadata
    metadata.append(item)
    num_patients += 1
    print(f"Downloaded patient {num_patients}/{max_patients}: {uid} from {institution}")


# Convert metadata to DataFrame
if len(metadata) > 0:
    df = pd.DataFrame(metadata)

    for institution in df["Institution"].unique():
        # Load metadata
        df_inst = df[df["Institution"] == institution]

        # Save metadata to CSV files
        path_metadata = output_root / institution / dir_config[config]["metadata"]
        path_metadata.mkdir(parents=True, exist_ok=True)

        df_anno = df_inst.drop(columns=["Institution", "Split", "Fold"])
        df_anno.to_csv(path_metadata / "annotation.csv", index=False)

        df_split = df_inst[["UID", "Split", "Fold"]]
        df_split.to_csv(path_metadata / "split.csv", index=False)

    print("\nDataset streamed and saved successfully!")
    print(f"Downloaded {num_patients} patients from split '{target_split}'")
    print(f"Output directory: {output_root}")
    
    # Organize for ResNet if requested
    if organize_for_resnet:
        print(f"\n{'='*60}")
        print("Organizing data for ResNet training...")
        print(f"{'='*60}")
        
        # Import and run organization function
        try:
            # Get the directory of this script
            script_dir = Path(__file__).parent
            organize_script = script_dir / "organize_for_resnet.py"
            
            if organize_script.exists():
                # Run the organization script
                result = subprocess.run(
                    [sys.executable, str(organize_script),
                     "--odelia-root", str(output_root),
                     "--output-dir", str(resnet_output_dir),
                     "--config", config],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(result.stdout)
                    print("\n✓ Data organized for ResNet!")
                    print(f"ResNet data directory: {resnet_output_dir}")
                    print("\nYou can now use this directory in ResNet config:")
                    print(f'  "data_dir": "{resnet_output_dir.absolute()}"')
                else:
                    print("Warning: Organization script had errors:")
                    print(result.stderr)
            else:
                print(f"Warning: Organization script not found at {organize_script}")
                print("Skipping organization step.")
        except Exception as e:
            print(f"Warning: Error during organization: {e}")
            print("You can manually run: python organize_for_resnet.py")
else:
    print("No patients were downloaded.")


from pathlib import Path
from datasets import load_dataset
import torchio as tio 
import numpy as np 
import pandas as pd
from tqdm import tqdm

# This is the MODIFIED ODELIA CODE for small sample test - T

# --------------------- Settings ---------------------
repo_id = "ODELIA-AI/ODELIA-Challenge-2025"
config = "unilateral"  # "default" or "unilateral"
output_root = Path("./Odelia/data")


# Limit and label filtering
max_samples = 10
candidate_label_keys = [
    "Label", "label", "Target", "target",
    "Diagnosis", "diagnosis", "Pathology", "pathology",
    "Malignancy", "malignancy", "Lesion", "lesion", "Lesion_Type", "lesion_type"
]


def item_has_any_label_fields(item: dict, label_keys: list[str]) -> bool:
    for key in label_keys:
        if key in item:
            value = item[key]
            if value is not None and not pd.isna(value):
                return True
    return False

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
num_kept = 0
reached_limit = False
for split, split_dataset in dataset.items():
    if reached_limit:
        break
    print("-------- Start Download - Split: ", split, " --------")
    for item in tqdm(split_dataset, desc="Downloading"):  # Stream data one-by-one
        if not item_has_any_label_fields(item, candidate_label_keys):
            continue
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
        num_kept += 1
        if num_kept >= max_samples:
            reached_limit = True
            break



# Convert metadata to DataFrame
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


print("Dataset streamed and saved successfully!")
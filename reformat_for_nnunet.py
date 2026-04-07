#!/usr/bin/env python3
"""Reformat raw_all (pre, post1, post2) to nnU-Net inference structure"""

import os
import shutil
import json
from pathlib import Path

# Directories
raw_all_dir = Path("pipeline/test/raw_all")
output_dir = Path("pipeline/test/nnUnet_raw/Dataset001_BreastLesion/imagesTs")
output_dir.mkdir(parents=True, exist_ok=True)

# Channel mapping
channels = {
    "pre": "0000",
    "post1": "0001",
    "post2": "0002"
}

# Counter for case numbering
case_id = 1
case_classification = {}

# Get all unique cases from post1 classes
for class_dir in sorted((raw_all_dir / "post1").iterdir()):
    if not class_dir.is_dir():
        continue

    classification = class_dir.name  # "benign" or "malignant"
    print(f"Processing {classification}...")

    for post1_file in sorted(class_dir.glob("*.nii.gz")):
        filename = post1_file.name

        # Copy all three channels for this case
        for seq_type, channel in channels.items():
            src_file = raw_all_dir / seq_type / class_dir.name / filename

            if not src_file.exists():
                print(f"  WARNING: {src_file} not found")
                continue

            new_name = f"case_{case_id:04d}_{channel}.nii.gz"
            dest_path = output_dir / new_name

            shutil.copy2(src_file, dest_path)

        case_key = f"case_{case_id:04d}"
        case_classification[case_key] = classification
        print(f"  {filename} -> {case_key}_[0000,0001,0002] ({classification})")
        case_id += 1

# Save classification mapping as JSON
mapping_file = output_dir.parent / "case_classification.json"
with open(mapping_file, "w") as f:
    json.dump(case_classification, f, indent=2)

print(f"\nProcessed {case_id - 1} cases with 3 channels each")
print(f"Classification mapping saved to {mapping_file}")
print("Structure ready for nnU-Net inference!")

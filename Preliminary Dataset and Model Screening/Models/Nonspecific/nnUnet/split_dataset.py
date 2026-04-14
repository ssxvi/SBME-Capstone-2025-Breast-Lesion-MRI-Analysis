import os
import shutil
import random
from glob import glob

# -----------------------------
# Top-level nnUNet_raw directory (your current layout)
# -----------------------------
RAW_DIR = "/scratch/st-ilker-1/yfeng40/nnUNet/nnUNet_data/nnUNet_raw"

# Source folders (moved one level higher, NOT inside Dataset503_ISPY1)
SOURCE_DIR = os.path.join(RAW_DIR, "images_bias-corrected_resampled_zscored_nifti")
LABEL_SOURCE_DIR = os.path.join(RAW_DIR, "labelsTr_backup_before_resample")

# Output dataset folder (nnU-Net expects imagesTr/labelsTr/etc inside here)
DATASET_DIR = os.path.join(RAW_DIR, "Dataset503_ISPY1")

IMAGES_TR = os.path.join(DATASET_DIR, "imagesTr")
IMAGES_TS = os.path.join(DATASET_DIR, "imagesTs")
LABELS_TR = os.path.join(DATASET_DIR, "labelsTr")
LABELS_TS = os.path.join(DATASET_DIR, "labelsTs")  # keep if you want GT for test; nnU-Net doesn't require it

# -----------------------------
# Recreate folders cleanly (avoid duplicates)
# -----------------------------
for d in [IMAGES_TR, IMAGES_TS, LABELS_TR, LABELS_TS]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# -----------------------------
# Collect cases (folders) and keep ONLY those with labels
# -----------------------------
all_cases = sorted([c for c in os.listdir(SOURCE_DIR) if c.startswith("ISPY")])

cases_with_labels = []
missing_labels = []

for c in all_cases:
    label_path = os.path.join(LABEL_SOURCE_DIR, f"{c}.nii.gz")
    if os.path.exists(label_path):
        cases_with_labels.append(c)
    else:
        missing_labels.append(c)

print(f"Image cases found:   {len(all_cases)}")
print(f"Labeled cases found: {len(cases_with_labels)}")
print(f"Missing labels:      {len(missing_labels)}")
if missing_labels:
    print("Missing label examples:", missing_labels[:20])

# Optional: save missing label IDs for DHF documentation
missing_txt = os.path.join(DATASET_DIR, "missing_labels.txt")
with open(missing_txt, "w") as f:
    for c in missing_labels:
        f.write(c + "\n")
print(f"Saved missing label list to: {missing_txt}")

cases = cases_with_labels

# -----------------------------
# Shuffle & Split (90/10)
# -----------------------------
random.seed(42)
random.shuffle(cases)

split_index = int(len(cases) * 0.9)
train_cases = cases[:split_index]
test_cases = cases[split_index:]

print(f"Total labeled cases: {len(cases)}")
print(f"Train cases:         {len(train_cases)}")
print(f"Test cases:          {len(test_cases)}")

# -----------------------------
# Stage one case: copy 3 channels (renamed) + copy label
# -----------------------------
def stage_case(case_id: str, img_dst: str, lbl_dst: str):
    case_path = os.path.join(SOURCE_DIR, case_id)

    ch_files = sorted(glob(os.path.join(case_path, "*.nii.gz")))
    if len(ch_files) != 3:
        raise ValueError(
            f"{case_id}: expected 3 channels, found {len(ch_files)}:\n{ch_files}"
        )

    # Copy + rename channels to nnU-Net format: CASE_0000/0001/0002
    for ch_idx, src in enumerate(ch_files):
        dst = os.path.join(img_dst, f"{case_id}_{ch_idx:04d}.nii.gz")
        shutil.copy2(src, dst)

    # Copy label (must exist because we filtered above)
    label_src = os.path.join(LABEL_SOURCE_DIR, f"{case_id}.nii.gz")
    shutil.copy2(label_src, os.path.join(lbl_dst, f"{case_id}.nii.gz"))

# -----------------------------
# Stage train (with progress)
# -----------------------------
for i, c in enumerate(train_cases, 1):
    stage_case(c, IMAGES_TR, LABELS_TR)
    if i % 10 == 0 or i == len(train_cases):
        print(f"Copied train {i}/{len(train_cases)}")

# -----------------------------
# Stage test (with progress)
# -----------------------------
for i, c in enumerate(test_cases, 1):
    stage_case(c, IMAGES_TS, LABELS_TS)
    print(f"Copied test {i}/{len(test_cases)}")

print("Done. Created imagesTr/imagesTs with _0000/_0001/_0002 and matching labelsTr/labelsTs.")
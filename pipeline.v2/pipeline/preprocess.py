"""
pipeline/preprocess.py
----------------------
Preprocessing for EfficientNet MIP inputs.
Refactored from newprecomputemips.py to be importable by the pipeline orchestrator.

Produces a (4, 256, 256) float32 numpy array per patient:
  channel 0 : MIP of (Post1 - Pre)
  channel 1 : MIP of (Post2 - Pre)
  channel 2 : MIP of (Post2 - Post1)
  channel 3 : MIP of Post2 (raw, joint-normalized)
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from collections import Counter
from multiprocessing import Pool

import cv2
import nibabel as nib
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE   = 256
TIMEPOINTS = ["Pre", "Post_1", "Post_2"]

TIMEPOINT_DIRS = {
    "Pre":    "pre",
    "Post_1": "post1",
    "Post_2": "post2",
}

LABEL_MAP = {
    "no_lesion": 0,
    "benign":    1,
    "malignant": 2,
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dicom_volume(folder_path: str | Path) -> np.ndarray:
    slices = []
    for file in sorted(Path(folder_path).glob("*.dcm")):
        ds = pydicom.dcmread(str(file))
        slices.append(ds.pixel_array)
    volume = np.stack(slices, axis=-1)
    return volume.astype(np.float32)


def load_nifti_volume(file_path: str | Path) -> np.ndarray:
    nii = nib.load(str(file_path))
    return nii.get_fdata().astype(np.float32)


def load_volume(path: str | Path) -> np.ndarray:
    path = str(path)
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        return load_nifti_volume(path)
    return load_dicom_volume(path)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(volume: np.ndarray, mode: str = "z") -> np.ndarray:
    if mode == "z":
        return (volume - volume.mean()) / (volume.std() + 1e-6)
    elif mode == "minmax":
        mn, mx = volume.min(), volume.max()
        return (volume - mn) / (mx - mn + 1e-6)
    raise ValueError(f"Unknown normalization mode: {mode}")


def joint_normalize(*volumes: np.ndarray) -> list[np.ndarray]:
    """
    Joint z-score normalization across all timepoints for a single patient.
    Uses the 5th–99th percentile of the combined distribution to compute
    mean/std, reducing the influence of enhancement outliers.
    """
    combined = np.concatenate([v.ravel() for v in volumes])
    low, high = np.percentile(combined, [5, 99])
    masked = combined[(combined > low) & (combined < high)]
    mean = masked.mean()
    std  = masked.std() + 1e-6
    return [(v - mean) / std for v in volumes]


# ---------------------------------------------------------------------------
# MIP construction
# ---------------------------------------------------------------------------

def resize_mip(mip: np.ndarray) -> np.ndarray:
    return cv2.resize(mip, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def topk_mean_projection(volume: np.ndarray, k: float = 0.25, axis: int = 2) -> np.ndarray:
    """Alternative to max-MIP: mean of top-k brightest slices along axis."""
    n_slices = volume.shape[axis]
    top_k    = max(1, int(n_slices * k))
    sorted_vol  = np.sort(volume, axis=axis)
    top_slices  = np.take(sorted_vol, range(n_slices - top_k, n_slices), axis=axis)
    return np.mean(top_slices, axis=axis)


# ---------------------------------------------------------------------------
# Core patient processor
# ---------------------------------------------------------------------------

def compute_mip_array(tp_dict: dict[str, str]) -> np.ndarray:
    """
    Given a dict of {timepoint: path}, load volumes, joint-normalize,
    compute subtraction MIPs, and return a (4, IMG_SIZE, IMG_SIZE) array.

    This is the main entry-point used by the pipeline orchestrator at
    inference time (single patient, no label needed).
    """
    vols_raw = {tp: load_volume(tp_dict[tp]) for tp in TIMEPOINTS}

    pre_n, post1_n, post2_n = joint_normalize(
        vols_raw["Pre"], vols_raw["Post_1"], vols_raw["Post_2"]
    )

    sub_post1_pre   = post1_n - pre_n
    sub_post2_pre   = post2_n - pre_n
    sub_post2_post1 = post2_n - post1_n

    channels = [
        np.max(sub_post1_pre,   axis=2),  # ch 0
        np.max(sub_post2_pre,   axis=2),  # ch 1
        np.max(sub_post2_post1, axis=2),  # ch 2
        np.max(post2_n,         axis=2),  # ch 3
    ]
    channels = [resize_mip(c) for c in channels]
    return np.stack(channels, axis=0).astype(np.float32)


def process_patient(tp_dict: dict[str, str], out_path: str | Path) -> None:
    """Load, process, and save a single patient's MIP array to disk."""
    stacked = compute_mip_array(tp_dict)
    np.save(str(out_path), stacked)
    logger.debug(f"Saved MIP cache → {out_path}")


# ---------------------------------------------------------------------------
# Dataset-level helpers (batch precomputation / split building)
# ---------------------------------------------------------------------------

def collect_samples(raw_root: str | Path) -> list[tuple[dict, int]]:
    """
    Walk raw_root and return [(tp_dict, label), ...] for every complete case.
    A case is skipped if any timepoint file is missing.
    """
    raw_root = Path(raw_root)
    pre_root = raw_root / TIMEPOINT_DIRS["Pre"]
    all_samples = []

    for class_name, label in LABEL_MAP.items():
        pre_class_dir = pre_root / class_name
        if not pre_class_dir.exists():
            logger.warning(f"Skipping missing class folder: {pre_class_dir}")
            continue

        for pre_file in sorted(pre_class_dir.glob("*.nii.gz")):
            filename = pre_file.name
            tp_dict = {
                tp: str(raw_root / TIMEPOINT_DIRS[tp] / class_name / filename)
                for tp in TIMEPOINTS
            }
            if all(Path(p).exists() for p in tp_dict.values()):
                all_samples.append((tp_dict, label))
            else:
                logger.warning(f"Skipping incomplete case (missing timepoint): {filename}")

    return all_samples


def _process_one(args: tuple) -> tuple[str, int]:
    tp_dict, label, out_file = args
    if not os.path.exists(out_file):
        process_patient(tp_dict, out_file)
    return (out_file, label)


def precompute_dataset(
    raw_root:    str | Path,
    output_root: str | Path,
    n_workers:   int = 4,
) -> list[tuple[str, int]]:
    """
    Precompute MIP .npy files for all patients in raw_root.
    Returns list of (out_file, label) tuples (same order as collect_samples).
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_samples = collect_samples(raw_root)
    if not all_samples:
        raise RuntimeError(f"No valid samples found under {raw_root}")

    logger.info(f"Found {len(all_samples)} samples. Processing with {n_workers} workers…")

    tasks = []
    for tp_dict, label in all_samples:
        patient_id = Path(tp_dict["Pre"]).name.replace(".nii.gz", "")
        out_file   = str(output_root / f"{patient_id}.npy")
        tasks.append((tp_dict, label, out_file))

    with Pool(n_workers) as pool:
        results = pool.map(_process_one, tasks)

    logger.info("MIP precomputation complete.")
    return results


# ---------------------------------------------------------------------------
# Label converters
# ---------------------------------------------------------------------------

def to_lesion_binary(label: int) -> int | None:
    return 0 if label == 0 else 1

def to_malig_benign(label: int) -> int | None:
    if label == 2:   return 1
    if label == 1:   return 0
    return None  # no_lesion cases excluded

def to_3class(label: int) -> int:
    return label


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------

def make_splits(
    all_results: list[tuple[str, int]],
    val_frac:    float,
    test_frac:   float,
    seed:        int,
) -> dict:
    paths  = [r[0] for r in all_results]
    labels = [r[1] for r in all_results]

    if test_frac > 0:
        paths_tv, paths_test, labels_tv, labels_test = train_test_split(
            paths, labels,
            test_size=test_frac, random_state=seed, stratify=labels,
        )
        val_frac_adj = val_frac / (1.0 - test_frac)
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            paths_tv, labels_tv,
            test_size=val_frac_adj, random_state=seed, stratify=labels_tv,
        )
        splits = {
            "train": [[p, l] for p, l in zip(paths_train, labels_train)],
            "val":   [[p, l] for p, l in zip(paths_val,   labels_val)],
            "test":  [[p, l] for p, l in zip(paths_test,  labels_test)],
        }
    else:
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            paths, labels,
            test_size=val_frac, random_state=seed, stratify=labels,
        )
        splits = {
            "train": [[p, l] for p, l in zip(paths_train, labels_train)],
            "val":   [[p, l] for p, l in zip(paths_val,   labels_val)],
        }

    for split_name, entries in splits.items():
        counts = Counter(e[1] for e in entries)
        logger.info(f"  {split_name}: {len(entries)} samples  {dict(sorted(counts.items()))}")

    return splits


def build_json(
    all_results: list[tuple[str, int]],
    label_fn,
    name:      str,
    json_root: str | Path,
    val_frac:  float = 0.1,
    test_frac: float = 0.1,
    seed:      int   = 42,
) -> None:
    json_root = Path(json_root)
    json_root.mkdir(parents=True, exist_ok=True)

    filtered = [
        (path, label_fn(label))
        for path, label in all_results
        if label_fn(label) is not None
    ]

    splits   = make_splits(filtered, val_frac, test_frac, seed)
    out_file = json_root / f"{name}.json"
    with open(out_file, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved split JSON → {out_file}")


# ---------------------------------------------------------------------------
# CLI entry-point (batch precomputation only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    BASE_DIR    = Path(__file__).resolve().parents[2]
    RAW_ROOT    = BASE_DIR / "test" / "raw_all"
    OUTPUT_ROOT = BASE_DIR / "test" / "mip_cache"
    JSON_ROOT   = BASE_DIR / "test" / "jsons"

    parser = argparse.ArgumentParser(description="Precompute MIP cache and split JSONs")
    parser.add_argument("--raw-root",    default=str(RAW_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--json-root",   default=str(JSON_ROOT))
    parser.add_argument("--workers",     type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--val-frac",    type=float, default=0.1)
    parser.add_argument("--test-frac",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    results = precompute_dataset(args.raw_root, args.output_root, args.workers)

    build_json(results, to_3class,        "combined_3class",       args.json_root, args.val_frac, args.test_frac, args.seed)
    build_json(results, to_lesion_binary, "combined_binary",       args.json_root, args.val_frac, args.test_frac, args.seed)
    build_json(results, to_malig_benign,  "combined_malig_benign", args.json_root, args.val_frac, args.test_frac, args.seed)

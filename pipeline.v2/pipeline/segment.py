"""
pipeline/segment.py
-------------------
Stage 3: Lesion segmentation via nnUNet (nnUNetv2_predict CLI).

nnUNet requires a very specific folder and filename convention.
This module handles:
  - Staging the input NIfTI files into nnUNet's expected imagesTs/ structure
  - Calling nnUNetv2_predict as a subprocess
  - Loading and returning the output segmentation mask

Expected nnUNet environment variables (set before running):
  nnUNet_raw       — path to nnUNet raw data root
  nnUNet_preprocessed — path to preprocessed data root
  nnUNet_results   — path where trained model weights live

nnUNet file naming convention:
  Input  : <case_id>_0000.nii.gz  (one file per modality/channel)
  Output : <case_id>.nii.gz       (integer label mask)

For a 3-timepoint DCE-MRI study mapped to a single 4-channel input,
all channels must be provided as separate _000N.nii.gz files.
nnUNet will assemble them according to its dataset.json configuration.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (override via environment variables or direct arguments)
# ---------------------------------------------------------------------------

NNUNET_RESULTS      = os.environ.get("nnUNet_results",      "")
NNUNET_DATASET_ID   = os.environ.get("NNUNET_DATASET_ID",   "001")
NNUNET_CONFIGURATION = os.environ.get("NNUNET_CONFIGURATION", "3d_fullres")
NNUNET_TRAINER      = os.environ.get("NNUNET_TRAINER",      "nnUNetTrainer")
NNUNET_FOLD         = os.environ.get("NNUNET_FOLD",         "0")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    mask: np.ndarray              # integer label array, same spatial shape as input volume
    mask_path: str                # path to the saved output .nii.gz mask
    affine: np.ndarray = field(default_factory=lambda: np.eye(4))
    case_id: str = ""

    @property
    def has_lesion(self) -> bool:
        """True if any non-background voxel is present in the mask."""
        return bool((self.mask > 0).any())

    @property
    def lesion_volume_voxels(self) -> int:
        return int((self.mask > 0).sum())

    def __str__(self) -> str:
        if self.has_lesion:
            return f"Segmentation: lesion found, {self.lesion_volume_voxels} voxels"
        return "Segmentation: no lesion voxels in mask"


# ---------------------------------------------------------------------------
# nnUNet file-staging helpers
# ---------------------------------------------------------------------------

def stage_input_volumes(
    tp_dict: dict[str, str],
    staging_dir: Path,
    case_id: str,
) -> list[Path]:
    """
    Copy each timepoint volume into the nnUNet imagesTs directory with
    the correct _000N.nii.gz naming convention.

    Mapping (adjust to match your dataset.json channel_names):
      Pre     → <case_id>_0000.nii.gz
      Post_1  → <case_id>_0001.nii.gz
      Post_2  → <case_id>_0002.nii.gz

    Returns list of staged file paths.
    """
    images_dir = staging_dir / "imagesTs"
    images_dir.mkdir(parents=True, exist_ok=True)

    channel_map = {
        "Pre":    0,
        "Post_1": 1,
        "Post_2": 2,
    }

    staged = []
    for tp_name, channel_idx in channel_map.items():
        src  = Path(tp_dict[tp_name])
        dest = images_dir / f"{case_id}_{channel_idx:04d}.nii.gz"
        shutil.copy2(str(src), str(dest))
        logger.debug(f"Staged {src.name} → {dest.name}")
        staged.append(dest)

    return staged


# ---------------------------------------------------------------------------
# nnUNet CLI runner
# ---------------------------------------------------------------------------

def run_nnunet_predict(
    input_dir:     Path,
    output_dir:    Path,
    dataset_id:    str = NNUNET_DATASET_ID,
    configuration: str = NNUNET_CONFIGURATION,
    trainer:       str = NNUNET_TRAINER,
    fold:          str = NNUNET_FOLD,
    extra_flags:   list[str] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run nnUNetv2_predict as a subprocess.

    nnUNet must be installed in the active Python environment (pip install nnunetv2).
    All nnUNet_* environment variables must be set before calling this function.

    Parameters
    ----------
    input_dir     : directory containing *_000N.nii.gz files
    output_dir    : directory where nnUNet writes the predicted mask
    dataset_id    : e.g. "001" for Dataset001_*
    configuration : "2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"
    trainer       : usually "nnUNetTrainer"
    fold          : "0"–"4" or "all"
    extra_flags   : any additional CLI flags, e.g. ["--save_probabilities"]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "nnunet_cpu_wrapper.py"),
        "-i",  str(input_dir),
        "-o",  str(output_dir),
        "-d",  dataset_id,
        "-c",  configuration,
        "-tr", trainer,
        "-f",  fold,
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    logger.info(f"Running nnUNet: {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["OMP_NUM_THREADS"] = "1"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        logger.error(f"nnUNet stderr:\n{result.stderr}")
        raise RuntimeError(
            f"nnUNetv2_predict failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    logger.info("nnUNet prediction complete.")
    logger.debug(f"nnUNet stdout:\n{result.stdout}")
    return result


# ---------------------------------------------------------------------------
# Mask loader
# ---------------------------------------------------------------------------

def load_mask(mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI segmentation mask; returns (array, affine)."""
    nii = nib.load(str(mask_path))
    return nii.get_fdata().astype(np.int32), nii.affine


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_segmentation(
    tp_dict:       dict[str, str],
    case_id:       str,
    output_dir:    str | Path,
    dataset_id:    str = NNUNET_DATASET_ID,
    configuration: str = NNUNET_CONFIGURATION,
    trainer:       str = NNUNET_TRAINER,
    fold:          str = NNUNET_FOLD,
    keep_staging:  bool = False,
) -> SegmentationResult:
    """
    Full segmentation pipeline for a single patient:
      1. Stage input volumes into nnUNet imagesTs/ structure
      2. Run nnUNetv2_predict
      3. Load and return the segmentation mask

    Parameters
    ----------
    tp_dict      : {"Pre": path, "Post_1": path, "Post_2": path}
    case_id      : unique identifier string (used for file naming)
    output_dir   : directory where the final mask .nii.gz is saved
    keep_staging : if True, the temporary staging directory is not deleted
                   (useful for debugging)

    Returns
    -------
    SegmentationResult
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    staging_ctx = (
        tempfile.TemporaryDirectory(prefix="nnunet_stage_")
        if not keep_staging
        else _PersistentDir(output_dir / "staging")
    )

    with staging_ctx as staging_str:
        staging_dir = Path(staging_str)
        images_dir  = staging_dir / "imagesTs"
        preds_dir   = staging_dir / "predictions"

        stage_input_volumes(tp_dict, staging_dir, case_id)
        run_nnunet_predict(
            input_dir     = images_dir,
            output_dir    = preds_dir,
            dataset_id    = dataset_id,
            configuration = configuration,
            trainer       = trainer,
            fold          = fold,
        )

        # nnUNet writes <case_id>.nii.gz (no _0000 suffix)
        pred_file = preds_dir / f"{case_id}.nii.gz"
        if not pred_file.exists():
            # Fallback: grab the first .nii.gz in preds_dir
            candidates = list(preds_dir.glob("*.nii.gz"))
            if not candidates:
                raise FileNotFoundError(
                    f"nnUNet produced no output in {preds_dir}. "
                    "Check nnUNet logs above."
                )
            pred_file = candidates[0]
            logger.warning(f"Expected {case_id}.nii.gz, using {pred_file.name} instead.")

        # Copy mask to persistent output directory
        final_mask_path = output_dir / f"{case_id}_seg.nii.gz"
        shutil.copy2(str(pred_file), str(final_mask_path))

        mask, affine = load_mask(final_mask_path)

    return SegmentationResult(
        mask      = mask,
        mask_path = str(final_mask_path),
        affine    = affine,
        case_id   = case_id,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

class _PersistentDir:
    """Context manager that creates a directory but does NOT delete it on exit."""
    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, *args):
        pass

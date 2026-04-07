"""
pipeline/run_pipeline.py
------------------------
Orchestrator: chains all pipeline stages for a single patient case.

Stages
------
1. Preprocess     — compute 4-channel MIP array from raw timepoint volumes
2. Lesion detect  — EfficientNet binary classifier (lesion vs no-lesion)
3. Segmentation   — nnUNet (only if lesion detected)
4. Malignancy     — EfficientNet binary classifier (only if lesion detected)
5. Report         — render HTML report with embedded figures

Usage (as a library)
--------------------
from pipeline.run_pipeline import run_pipeline, PipelineConfig

cfg    = PipelineConfig(case_id="patient_001", tp_dict={...}, output_dir="./results")
result = run_pipeline(cfg)
print(result.report.overall_impression)

Usage (CLI)
-----------
python -m pipeline.run_pipeline \
    --case-id patient_001 \
    --pre   /data/raw/pre/patient_001.nii.gz \
    --post1 /data/raw/post1/patient_001.nii.gz \
    --post2 /data/raw/post2/patient_001.nii.gz \
    --output-dir ./results
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .preprocess          import compute_mip_array
from .classify_lesion     import LesionResult,     classify_lesion
from .segment             import SegmentationResult, run_segmentation
from .classify_malignancy import MalignancyResult,  classify_malignancy
from .report              import PipelineReport,    generate_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    case_id:    str
    tp_dict:    dict[str, str]        # {"Pre": path, "Post_1": path, "Post_2": path}
    output_dir: str | Path = "./results"

    # Model weights (None → use env vars / defaults)
    lesion_weights_path:     Optional[str] = None
    malignancy_weights_path: Optional[str] = None

    # nnUNet settings
    nnunet_dataset_id:    str = "001"
    nnunet_configuration: str = "3d_fullres"
    nnunet_trainer:       str = "nnUNetTrainer"
    nnunet_fold:          str = "0"

    # Behaviour flags
    skip_segmentation:   bool = False   # set True to run classify-only (no nnUNet)
    force_segmentation:  bool = False   # set True to always run segmentation regardless of lesion threshold
    use_mask_for_roi:    bool = True    # crop MIP to lesion bbox for malignancy step
    lesion_threshold:    float = 0.5    # p(lesion) threshold for conditional branching
    malignancy_margin:   int  = 16      # ROI crop margin in pixels

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-run result bundle
# ---------------------------------------------------------------------------

@dataclass
class PipelineRunResult:
    config:       PipelineConfig
    mip_array:    Optional[np.ndarray]        = None
    lesion:       Optional[LesionResult]      = None
    segmentation: Optional[SegmentationResult] = None
    malignancy:   Optional[MalignancyResult]  = None
    report:       Optional[PipelineReport]    = None
    report_html:  str                         = ""
    report_path:  str                         = ""
    success:      bool                        = True
    error:        str                         = ""


# ---------------------------------------------------------------------------
# Stage runners (each logs its own header / footer)
# ---------------------------------------------------------------------------

def _stage_preprocess(cfg: PipelineConfig) -> np.ndarray:
    logger.info(f"[{cfg.case_id}] Stage 1/5 — Preprocessing")
    mip = compute_mip_array(cfg.tp_dict)
    logger.info(f"[{cfg.case_id}] MIP array shape: {mip.shape}")
    return mip


def _stage_classify_lesion(cfg: PipelineConfig, mip: np.ndarray) -> LesionResult:
    logger.info(f"[{cfg.case_id}] Stage 2/5 — Lesion classification")
    result = classify_lesion(mip, weights_path=cfg.lesion_weights_path)
    logger.info(f"[{cfg.case_id}] → {result}")
    return result


def _stage_segmentation(cfg: PipelineConfig) -> SegmentationResult:
    logger.info(f"[{cfg.case_id}] Stage 3/5 — Segmentation (nnUNet)")
    seg_dir = cfg.output_dir / "segmentations"
    result  = run_segmentation(
        tp_dict       = cfg.tp_dict,
        case_id       = cfg.case_id,
        output_dir    = seg_dir,
        dataset_id    = cfg.nnunet_dataset_id,
        configuration = cfg.nnunet_configuration,
        trainer       = cfg.nnunet_trainer,
        fold          = cfg.nnunet_fold,
    )
    logger.info(f"[{cfg.case_id}] → {result}")
    return result


def _stage_classify_malignancy(
    cfg: PipelineConfig,
    mip: np.ndarray,
    seg: Optional[SegmentationResult],
) -> MalignancyResult:
    logger.info(f"[{cfg.case_id}] Stage 4/5 — Malignancy classification")
    mask_3d = seg.mask if (seg is not None and cfg.use_mask_for_roi) else None
    result  = classify_malignancy(
        mip_array    = mip,
        mask_3d      = mask_3d,
        weights_path = cfg.malignancy_weights_path,
        margin       = cfg.malignancy_margin,
    )
    logger.info(f"[{cfg.case_id}] → {result}")
    return result


def _stage_report(
    cfg:         PipelineConfig,
    mip:         Optional[np.ndarray],
    lesion:      Optional[LesionResult],
    seg:         Optional[SegmentationResult],
    malignancy:  Optional[MalignancyResult],
    error_msg:   str = "",
) -> tuple[PipelineReport, str, str]:
    logger.info(f"[{cfg.case_id}] Stage 5/5 — Report generation")

    report = PipelineReport(case_id=cfg.case_id)

    if lesion is not None:
        report.lesion_detected      = lesion.lesion_detected
        report.lesion_confidence    = lesion.confidence
        report.lesion_probabilities = lesion.probabilities

    if seg is not None:
        report.segmentation_run        = True
        report.segmentation_mask_path  = seg.mask_path
        report.lesion_volume_voxels    = seg.lesion_volume_voxels

    if malignancy is not None:
        report.malignancy_label          = malignancy.label_name
        report.malignancy_confidence     = malignancy.confidence
        report.malignancy_probabilities  = malignancy.probabilities

    if error_msg:
        report.error_message = error_msg

    mask_3d = seg.mask if seg is not None else None
    html, path = generate_report(
        report     = report,
        mip_array  = mip,
        mask_3d    = mask_3d,
        output_dir = cfg.output_dir,
    )
    return report, html, path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(cfg: PipelineConfig) -> PipelineRunResult:
    """
    Run all pipeline stages for a single patient.

    Conditional logic
    -----------------
    - Segmentation and malignancy stages are SKIPPED if the lesion classifier
      returns p(lesion) < cfg.lesion_threshold.
    - If cfg.skip_segmentation is True, nnUNet is bypassed entirely and the
      malignancy classifier runs on the full-FOV MIP.
    - Any stage exception is caught, logged, and surfaced in the report —
      the run returns success=False but still produces a partial report.
    """
    run = PipelineRunResult(config=cfg)
    error_msg = ""

    try:
        # Stage 1 — Preprocess
        run.mip_array = _stage_preprocess(cfg)

        # Stage 2 — Lesion detection
        run.lesion = _stage_classify_lesion(cfg, run.mip_array)

        lesion_prob = run.lesion.probabilities[1]  # p(lesion)
        lesion_confirmed = lesion_prob >= cfg.lesion_threshold

        if lesion_confirmed or cfg.force_segmentation:
            # Stage 3 — Segmentation
            if not cfg.skip_segmentation:
                try:
                    run.segmentation = _stage_segmentation(cfg)
                except Exception as exc:
                    logger.warning(
                        f"[{cfg.case_id}] Segmentation failed, continuing without mask: {exc}"
                    )

            # Stage 4 — Malignancy
            run.malignancy = _stage_classify_malignancy(cfg, run.mip_array, run.segmentation)
        else:
            logger.info(
                f"[{cfg.case_id}] No lesion detected "
                f"(p={lesion_prob:.3f} < threshold={cfg.lesion_threshold}). "
                "Skipping segmentation and malignancy stages."
            )

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error(f"[{cfg.case_id}] Pipeline error:\n{error_msg}")
        run.success = False
        run.error   = str(exc)

    # Stage 5 — Report (always runs, even after errors)
    try:
        run.report, run.report_html, run.report_path = _stage_report(
            cfg        = cfg,
            mip        = run.mip_array,
            lesion     = run.lesion,
            seg        = run.segmentation,
            malignancy = run.malignancy,
            error_msg  = error_msg,
        )
        logger.info(f"[{cfg.case_id}] Pipeline complete → {run.report.overall_impression}")
    except Exception as exc:
        logger.error(f"[{cfg.case_id}] Report generation failed: {exc}")

    return run


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the full MRI analysis pipeline")
    parser.add_argument("--case-id",    required=True)
    parser.add_argument("--pre",        required=True, help="Path to Pre timepoint .nii.gz")
    parser.add_argument("--post1",      required=True, help="Path to Post_1 timepoint .nii.gz")
    parser.add_argument("--post2",      required=True, help="Path to Post_2 timepoint .nii.gz")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--skip-seg",   action="store_true", help="Skip nnUNet segmentation")
    parser.add_argument("--force-seg",  action="store_true", help="Always run segmentation regardless of lesion threshold")
    parser.add_argument("--lesion-weights",     default=None)
    parser.add_argument("--malignancy-weights", default=None)
    args = parser.parse_args()

    cfg = PipelineConfig(
        case_id    = args.case_id,
        tp_dict    = {"Pre": args.pre, "Post_1": args.post1, "Post_2": args.post2},
        output_dir = args.output_dir,
        skip_segmentation       = args.skip_seg,
        force_segmentation      = args.force_seg,
        lesion_weights_path     = args.lesion_weights,
        malignancy_weights_path = args.malignancy_weights,
    )

    result = run_pipeline(cfg)

    if result.report_path:
        print(f"\nReport: {result.report_path}")
    if not result.success:
        import sys
        sys.exit(1)

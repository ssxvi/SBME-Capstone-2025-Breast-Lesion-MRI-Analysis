"""
api/main.py
-----------
FastAPI application exposing the MRI analysis pipeline.

Endpoints
---------
GET  /health                   — liveness check
POST /upload                   — upload a single .nii.gz timepoint file
POST /run                      — submit a pipeline job (returns run_id immediately)
GET  /result/{run_id}          — poll for results
GET  /report/{run_id}          — serve the HTML report
DELETE /runs/{run_id}          — clean up a completed run

Design notes
------------
- Jobs run in a ThreadPoolExecutor (CPU-only; swap for ProcessPoolExecutor if
  you add GPU workers later). Each job gets a UUID run_id.
- Results are stored in an in-process dict (replace with Redis/DB for prod).
- CORS is open for local dev; lock down origins before deployment.
"""

from __future__ import annotations

import logging
import os
import time
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import re

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (
    HealthResponse,
    UploadResponse,
    RunPipelineRequest,
    PipelineRunResponse,
    PipelineResultResponse,
    PipelineStatus,
    LesionResult     as LesionSchema,
    SegmentationResult as SegSchema,
    MalignancyResult as MalignancySchema,
    LesionLabel,
    MalignancyLabel,
)

# Pipeline imports (resolved relative to project root)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.run_pipeline import PipelineConfig, PipelineRunResult, run_pipeline
from pipeline.classify_lesion import LesionResult as LesionRuntime
from pipeline.classify_malignancy import MalignancyResult as MalignancyRuntime
from pipeline.segment import SegmentationResult as SegmentationRuntime
from pipeline.report import PipelineReport, generate_report
from pipeline.preprocess import compute_mip_array

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "MRI Analysis Pipeline API",
    description = "EfficientNet lesion detection & malignancy classification + nnUNet segmentation",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Restrict in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parents[1]
UPLOAD_DIR  = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MOCK_SEGMENTATION_DIR = Path(os.environ.get("PIPELINE_MOCK_SEGMENTATION_DIR", str(BASE_DIR / "mock_segmentation")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MOCK_SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/mock-segmentation", StaticFiles(directory=str(MOCK_SEGMENTATION_DIR)), name="mock_segmentation")
app.mount("/results-static", StaticFiles(directory=str(RESULTS_DIR)), name="results_static")

# ---------------------------------------------------------------------------
# In-process job store  (replace with Redis/DB in production)
# ---------------------------------------------------------------------------

_job_store: dict[str, PipelineRunResult | dict] = {}
_executor  = ThreadPoolExecutor(max_workers=int(os.environ.get("PIPELINE_WORKERS", 2)))


def _safe_filename(value: str) -> str:
    """Convert arbitrary labels (e.g., case_id) into filesystem-safe names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "case"


def _build_demo_mask(height: int = 256, width: int = 256, depth: int = 24) -> np.ndarray:
    """Create a compact synthetic lesion mask for accelerated responses."""
    # Prefer NIfTI demo mask so demo mode mirrors production mask format.
    for nifti_name in ["demo_mask.nii.gz", "demo_mask.nii"]:
        mask_nifti = MOCK_SEGMENTATION_DIR / nifti_name
        if mask_nifti.exists():
            loaded = nib.load(str(mask_nifti)).get_fdata()
            if loaded.ndim == 3:
                return (loaded > 0).astype(np.int32)

    # Backward-compatible fallback for older mock assets.
    mask_npy = MOCK_SEGMENTATION_DIR / "demo_mask.npy"
    if mask_npy.exists():
        loaded = np.load(mask_npy)
        if loaded.ndim == 3:
            return (loaded > 0).astype(np.int32)

    y, x = np.ogrid[:height, :width]
    center_y, center_x = int(height * 0.52), int(width * 0.58)
    radius_y, radius_x = int(height * 0.16), int(width * 0.11)
    mask_2d = (((y - center_y) / max(radius_y, 1)) ** 2 + ((x - center_x) / max(radius_x, 1)) ** 2) <= 1.0
    mask_3d = np.repeat(mask_2d[:, :, None], depth, axis=2)
    return mask_3d.astype(np.int32)


def _get_mock_preview_image_url() -> Optional[str]:
    for filename in ["segmentation_overlay.png", "segmentation_overlay.jpg", "segmentation_overlay.jpeg"]:
        candidate = MOCK_SEGMENTATION_DIR / filename
        if candidate.exists():
            return f"/mock-segmentation/{filename}"
    return None


def _write_center_slice_preview(mask: np.ndarray, out_path: Path) -> None:
    """Save a quick center-slice visualization for the demo segmentation mask."""
    if mask.ndim != 3:
        raise ValueError("Expected a 3D mask array for center-slice preview.")

    z = mask.shape[2] // 2
    center_slice = (mask[:, :, z] > 0).astype(np.float32)

    fig, ax = plt.subplots(figsize=(4, 4), facecolor="white")
    ax.imshow(center_slice, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title("Center Slice")
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _build_demo_report(
    cfg: PipelineConfig,
    lesion: LesionRuntime,
    seg: SegmentationRuntime,
    malignancy: MalignancyRuntime,
    mip_array: np.ndarray | None,
) -> tuple[PipelineReport, str, str]:
    """Build demo reports through the same generator/template as regular runs."""
    report = PipelineReport(case_id=cfg.case_id)
    report.lesion_detected = lesion.lesion_detected
    report.lesion_confidence = lesion.confidence
    report.lesion_probabilities = lesion.probabilities
    report.segmentation_run = True
    report.segmentation_mask_path = seg.mask_path
    report.lesion_volume_voxels = seg.lesion_volume_voxels
    report.lesion_volume_mm3 = seg.lesion_volume_mm3
    report.malignancy_label = malignancy.label_name
    report.malignancy_confidence = malignancy.confidence
    report.malignancy_probabilities = malignancy.probabilities

    demo_html_path = os.environ.get("PIPELINE_DEMO_REPORT_HTML", "").strip()
    if demo_html_path:
        p = Path(demo_html_path)
        if p.exists():
            html = p.read_text(encoding="utf-8")
            out_path = str(cfg.output_dir / f"{_safe_filename(cfg.case_id)}_report.html")
            Path(out_path).write_text(html, encoding="utf-8")
            return report, html, out_path

    return_report_html, return_report_path = generate_report(
        report=report,
        mip_array=mip_array,
        mask_3d=seg.mask,
        output_dir=cfg.output_dir,
    )
    return report, return_report_html, return_report_path


def _run_demo_job(run_id: str, cfg: PipelineConfig, duration_sec: float) -> None:
    """Simulate stage progress and produce a realistic final payload for demos."""
    entry = _job_store[run_id]
    entry["status"] = PipelineStatus.running

    stages = [
        "preprocessing",
        "lesion_classification",
        "segmentation",
        "malignancy_classification",
        "report_generation",
    ]

    total = max(float(duration_sec), 1.0)
    per_stage = total / len(stages)

    for i, stage in enumerate(stages):
        stage_start = i / len(stages)
        entry["current_stage"] = stage
        ticks = max(3, int(per_stage / 0.5))
        for t in range(ticks):
            frac = (t + 1) / ticks
            entry["progress"] = min(stage_start + (frac / len(stages)), 0.99)
            time.sleep(per_stage / ticks)

    mask = _build_demo_mask()
    area_px = int((mask > 0).max(axis=2).sum())
    volume_voxels = int((mask > 0).sum())
    safe_case_id = _safe_filename(cfg.case_id)

    seg_dir = cfg.output_dir / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    mask_path = seg_dir / f"{safe_case_id}_demo_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.int16), np.eye(4)), str(mask_path))

    preview_filename = f"{safe_case_id}_center_slice.png"
    preview_path = seg_dir / preview_filename
    _write_center_slice_preview(mask, preview_path)

    result = PipelineRunResult(
        config=cfg,
        success=True,
    )
    demo_mip_array: np.ndarray | None = None
    try:
        demo_mip_array = compute_mip_array(cfg.tp_dict)
    except Exception as exc:
        logger.warning(f"[{cfg.case_id}] Demo MIP generation failed, continuing without MIP figures: {exc}")

    result.lesion = LesionRuntime(label=1, confidence=0.93, probabilities=[0.07, 0.93])
    result.segmentation = SegmentationRuntime(mask=mask, mask_path=str(mask_path), case_id=cfg.case_id)
    result.malignancy = MalignancyRuntime(label=1, confidence=0.88, probabilities=[0.12, 0.88])
    result.report, result.report_html, result.report_path = _build_demo_report(
        cfg=cfg,
        lesion=result.lesion,
        seg=result.segmentation,
        malignancy=result.malignancy,
        mip_array=demo_mip_array,
    )

    entry["result"] = result
    entry["segmentation_preview_url"] = f"/results-static/{run_id}/segmentations/{preview_filename}"
    entry["current_stage"] = "complete"
    entry["progress"] = 1.0
    entry["status"] = PipelineStatus.complete

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(run_id: str, cfg: PipelineConfig, demo_duration_sec: float) -> None:
    """Execute the pipeline in a thread pool and store the result."""
    logger.info(f"Job {run_id} starting for case {cfg.case_id}")
    entry = _job_store[run_id]

    if entry.get("demo_mode"):
        logger.info(f"Job {run_id} running in demo mode")
        try:
            _run_demo_job(run_id, cfg, duration_sec=demo_duration_sec)
        except Exception as exc:
            entry["status"] = PipelineStatus.failed
            entry["current_stage"] = "failed"
            entry["progress"] = 1.0
            entry["result"] = PipelineRunResult(config=cfg, success=False, error=str(exc))
            logger.exception(f"Job {run_id} demo run failed: {exc}")
        return

    entry["status"] = PipelineStatus.running
    entry["current_stage"] = "full_pipeline"
    entry["progress"] = 0.2

    result = run_pipeline(cfg)

    entry["result"] = result
    entry["progress"] = 1.0
    entry["current_stage"] = "complete" if result.success else "failed"
    entry["status"] = PipelineStatus.complete if result.success else PipelineStatus.failed
    logger.info(f"Job {run_id} finished — {entry['status']}")


# ---------------------------------------------------------------------------
# Helper: build result response from a completed PipelineRunResult
# ---------------------------------------------------------------------------

def _build_result_response(run_id: str, entry: dict) -> PipelineResultResponse:
    status: PipelineStatus = entry["status"]
    case_id: str           = entry["case_id"]
    result: Optional[PipelineRunResult] = entry.get("result")
    current_stage: Optional[str] = entry.get("current_stage")
    progress: Optional[float] = entry.get("progress")
    segmentation_preview_url: Optional[str] = entry.get("segmentation_preview_url")

    if result is None:
        return PipelineResultResponse(
            run_id=run_id,
            case_id=case_id,
            status=status,
            current_stage=current_stage,
            progress=progress,
        )

    lesion_schema = None
    if result.lesion is not None:
        lesion_schema = LesionSchema(
            label         = LesionLabel.lesion if result.lesion.lesion_detected else LesionLabel.no_lesion,
            confidence    = result.lesion.confidence,
            probabilities = result.lesion.probabilities,
        )

    seg_schema = None
    if result.segmentation is not None:
        area_px = None
        if result.segmentation.mask is not None:
            area_px = int((result.segmentation.mask > 0).max(axis=2).sum())
        seg_schema = SegSchema(
            lesion_volume_voxels = result.segmentation.lesion_volume_voxels,
            lesion_area_pixels   = area_px,
            mask_path            = result.segmentation.mask_path,
            preview_image_url    = segmentation_preview_url,
        )

    malig_schema = None
    if result.malignancy is not None:
        malig_schema = MalignancySchema(
            label         = MalignancyLabel(result.malignancy.label_name),
            confidence    = result.malignancy.confidence,
            probabilities = result.malignancy.probabilities,
        )

    report_url = f"/report/{run_id}" if result.report_html else None

    return PipelineResultResponse(
        run_id             = run_id,
        case_id            = case_id,
        status             = status,
        current_stage      = current_stage,
        progress           = progress,
        error              = result.error or None,
        lesion             = lesion_schema,
        segmentation       = seg_schema,
        malignancy         = malig_schema,
        overall_impression = result.report.overall_impression if result.report else None,
        report_url         = report_url,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse()


@app.post("/upload", response_model=UploadResponse, tags=["Files"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a single .nii.gz timepoint file.
    Returns the server-side path to pass in /run requests.

    Call this three times (Pre, Post_1, Post_2) before submitting a run.
    """
    if not file.filename.endswith((".nii.gz", ".nii", ".dcm")):
        raise HTTPException(
            status_code=400,
            detail="Only .nii.gz, .nii, or .dcm files are accepted."
        )

    dest = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    content = await file.read()
    dest.write_bytes(content)

    logger.info(f"Uploaded {file.filename} → {dest} ({len(content)} bytes)")
    return UploadResponse(
        filename    = file.filename,
        server_path = str(dest),
        size_bytes  = len(content),
    )


@app.post("/run", response_model=PipelineRunResponse, tags=["Pipeline"])
async def submit_run(request: RunPipelineRequest, background_tasks: BackgroundTasks):
    """
    Submit a pipeline job. Returns a run_id immediately.
    Poll GET /result/{run_id} to check progress.
    """
    # Validate that all three input files exist on the server
    for label, path in [
        ("pre_path",   request.pre_path),
        ("post1_path", request.post1_path),
        ("post2_path", request.post2_path),
    ]:
        if not Path(path).exists():
            raise HTTPException(status_code=400, detail=f"File not found: {label} = {path}")

    run_id  = str(uuid.uuid4())
    out_dir = RESULTS_DIR / run_id

    cfg = PipelineConfig(
        case_id    = request.case_id,
        tp_dict    = {
            "Pre":    request.pre_path,
            "Post_1": request.post1_path,
            "Post_2": request.post2_path,
        },
        output_dir           = out_dir,
        skip_segmentation    = request.skip_segmentation,
        use_mask_for_roi     = request.use_mask_for_roi,
        lesion_threshold     = request.lesion_threshold,
        nnunet_dataset_id    = request.nnunet_dataset_id,
        nnunet_configuration = request.nnunet_configuration,
        nnunet_fold          = request.nnunet_fold,
    )

    _job_store[run_id] = {
        "status":        PipelineStatus.pending,
        "case_id":       request.case_id,
        "result":        None,
        "demo_mode":     request.demo_mode,
        "current_stage": "queued",
        "progress":      0.0,
    }

    _executor.submit(_run_job, run_id, cfg, request.demo_duration_sec)
    logger.info(f"Accepted job {run_id} for case {request.case_id}")

    return PipelineRunResponse(
        run_id  = run_id,
        case_id = request.case_id,
        status  = PipelineStatus.pending,
        demo_mode = request.demo_mode,
    )


@app.get("/result/{run_id}", response_model=PipelineResultResponse, tags=["Pipeline"])
def get_result(run_id: str):
    """
    Poll for the status and results of a pipeline run.
    Status transitions: pending → running → complete | failed
    """
    entry = _job_store.get(run_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return _build_result_response(run_id, entry)


@app.get("/report/{run_id}", response_class=HTMLResponse, tags=["Pipeline"])
def get_report(run_id: str):
    """Serve the generated HTML report for a completed run."""
    entry = _job_store.get(run_id)
    if entry is None:
        logger.warning(f"Report requested for unknown run_id: {run_id}")
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    result: Optional[PipelineRunResult] = entry.get("result")
    if result is None:
        logger.warning(f"Report requested but result is None for run_id: {run_id}")
        raise HTTPException(status_code=404, detail="Pipeline still running or failed before result was generated.")

    if not result.report_html:
        logger.warning(f"Report requested but report_html is empty for run_id: {run_id}. Status: {entry.get('status')}, Error: {result.error}")
        raise HTTPException(status_code=404, detail=f"Report not yet available. Pipeline status: {entry.get('status')}. Error: {result.error}")

    logger.info(f"Serving report for run_id: {run_id} ({len(result.report_html)} bytes)")
    return HTMLResponse(content=result.report_html)


@app.delete("/runs/{run_id}", tags=["System"])
def delete_run(run_id: str):
    """Remove a run from the job store and delete its output directory."""
    entry = _job_store.pop(run_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

    out_dir = RESULTS_DIR / run_id
    if out_dir.exists():
        shutil.rmtree(out_dir)

    return JSONResponse({"deleted": run_id})


# ---------------------------------------------------------------------------
# Dev server entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

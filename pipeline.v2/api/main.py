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
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# In-process job store  (replace with Redis/DB in production)
# ---------------------------------------------------------------------------

_job_store: dict[str, PipelineRunResult | dict] = {}
_executor  = ThreadPoolExecutor(max_workers=int(os.environ.get("PIPELINE_WORKERS", 2)))

# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(run_id: str, cfg: PipelineConfig) -> None:
    """Execute the pipeline in a thread pool and store the result."""
    logger.info(f"Job {run_id} starting for case {cfg.case_id}")
    _job_store[run_id]["status"] = PipelineStatus.running

    result = run_pipeline(cfg)

    _job_store[run_id]["result"] = result
    _job_store[run_id]["status"] = (
        PipelineStatus.complete if result.success else PipelineStatus.failed
    )
    logger.info(f"Job {run_id} finished — {_job_store[run_id]['status']}")


# ---------------------------------------------------------------------------
# Helper: build result response from a completed PipelineRunResult
# ---------------------------------------------------------------------------

def _build_result_response(run_id: str, entry: dict) -> PipelineResultResponse:
    status: PipelineStatus = entry["status"]
    case_id: str           = entry["case_id"]
    result: Optional[PipelineRunResult] = entry.get("result")

    if result is None:
        return PipelineResultResponse(run_id=run_id, case_id=case_id, status=status)

    lesion_schema = None
    if result.lesion is not None:
        lesion_schema = LesionSchema(
            label         = LesionLabel.lesion if result.lesion.lesion_detected else LesionLabel.no_lesion,
            confidence    = result.lesion.confidence,
            probabilities = result.lesion.probabilities,
        )

    seg_schema = None
    if result.segmentation is not None:
        seg_schema = SegSchema(
            lesion_volume_voxels = result.segmentation.lesion_volume_voxels,
            mask_path            = result.segmentation.mask_path,
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
        "status":  PipelineStatus.pending,
        "case_id": request.case_id,
        "result":  None,
    }

    _executor.submit(_run_job, run_id, cfg)
    logger.info(f"Accepted job {run_id} for case {request.case_id}")

    return PipelineRunResponse(
        run_id  = run_id,
        case_id = request.case_id,
        status  = PipelineStatus.pending,
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

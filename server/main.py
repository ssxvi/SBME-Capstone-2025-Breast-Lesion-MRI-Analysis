"""
Breast Lesion Pipeline API — local development server.
Run with:  uvicorn main:app --reload --port 8000
"""

import asyncio
import csv
import io
import json
import random
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Breast Lesion Pipeline API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Types / constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".nii", ".gz", ".dcm"}
STEP_KEYS = ["validate", "convert", "crop", "lesionDetection", "lesionType", "segmentation", "report"]

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

uploads: dict[str, dict] = {}
jobs:    dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateJobRequest(BaseModel):
    upload_id: str
    pipeline_name: str = Field(min_length=1, max_length=200)
    has_external_chest: bool = False
    report_format: Literal["csv", "json"] = "csv"
    notes: Optional[str] = None

class ValidateInputRequest(BaseModel):
    upload_id: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def detect_input_type(filenames: list[str]) -> str:
    for name in filenames:
        if name.endswith(".dcm"):
            return "dicom"
    return "nifti"

def validate_extensions(filenames: list[str]) -> list[str]:
    bad = []
    for name in filenames:
        if not any(name.endswith(ext) for ext in ALLOWED_EXTENSIONS):
            bad.append(name)
    return bad

def initial_steps() -> dict:
    return {k: "pending" for k in STEP_KEYS}

def build_result(job: dict) -> dict:
    upload     = uploads[job["upload_id"]]
    filenames  = [f["name"] for f in upload["files"]]
    input_type = upload["detected_input_type"]

    lesion_prob      = round(random.uniform(0.55, 0.95), 2)
    lesion_type_prob = round(random.uniform(0.60, 0.95), 2)
    lesion_label     = "Lesion" if lesion_prob >= 0.5 else "No Lesion"
    lesion_type      = "Malignant" if lesion_type_prob >= 0.5 else "Benign"
    dicom_to_nifti   = "performed" if input_type == "dicom" else "not_needed"
    has_chest        = job.get("has_external_chest", False)

    return {
        "job_id": job["job_id"],
        "status": "completed",
        "summary": {
            "lesionLabel":           lesion_label,
            "lesionProbability":     lesion_prob,
            "lesionTypeLabel":       lesion_type,
            "lesionTypeProbability": lesion_type_prob,
            "segmentationStatus":    "Mask generated",
            "maskFilename":          "predicted_mask.nii.gz",
        },
        "report_row": {
            "pipeline_name":                job["pipeline_name"],
            "uploaded_input_type":          input_type,
            "uploaded_files":               " | ".join(filenames),
            "external_chest_present":       "yes" if has_chest else "no",
            "dicom_to_nifti":               dicom_to_nifti,
            "manual_cropping":              "not_needed",
            "lesion_screening_label":       lesion_label,
            "lesion_screening_probability": f"{lesion_prob:.3f}",
            "lesion_type_label":            lesion_type,
            "lesion_type_probability":      f"{lesion_type_prob:.3f}",
            "segmentation_result":          "Mask generated",
            "segmentation_mask":            "predicted_mask.nii.gz",
            "notes":                        job.get("notes") or "",
        },
    }

# ---------------------------------------------------------------------------
# Background pipeline simulation
# ---------------------------------------------------------------------------

STEP_SEQUENCE = [
    # (step_key, duration_seconds, maybe_skip)
    ("validate",        1.0, False),
    ("convert",         2.0, True),   # skipped for nifti input
    ("crop",            1.5, True),   # randomly skipped ~50 %
    ("lesionDetection", 3.0, False),
    ("lesionType",      2.5, False),
    ("segmentation",    3.0, False),
    ("report",          1.0, False),
]

async def run_pipeline(job_id: str):
    job        = jobs[job_id]
    upload     = uploads[job["upload_id"]]
    input_type = upload["detected_input_type"]
    total      = len(STEP_SEQUENCE)

    job["status"] = "running"

    for i, (step_key, duration, maybe_skip) in enumerate(STEP_SEQUENCE):
        if job["status"] == "cancelled":
            return

        skip = False
        if maybe_skip:
            if step_key == "convert" and input_type == "nifti":
                skip = True
            elif step_key == "crop" and random.random() < 0.5:
                skip = True

        if skip:
            job["steps"][step_key] = "skipped"
        else:
            job["steps"][step_key] = "running"
            job["message"]         = f"Running step: {step_key}"
            job["updated_at"]      = now_iso()
            await asyncio.sleep(duration)

            if job["status"] == "cancelled":
                return

            job["steps"][step_key] = "complete"

        job["progress"]   = int(((i + 1) / total) * 100)
        job["updated_at"] = now_iso()

    job["status"]     = "completed"
    job["progress"]   = 100
    job["message"]    = "Pipeline complete."
    job["result"]     = build_result(job)
    job["updated_at"] = now_iso()

# ---------------------------------------------------------------------------
# Endpoint 9: Health
# ---------------------------------------------------------------------------

@app.get("/v1/health")
def health():
    return {"status": "ok", "service": "breast-lesion-pipeline-api", "version": "0.1.0"}

# ---------------------------------------------------------------------------
# Endpoint 1: Upload files
# ---------------------------------------------------------------------------

@app.post("/v1/uploads", status_code=201)
async def upload_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files provided.")

    file_records = []
    for f in files:
        filename = f.filename or ""
        content  = await f.read()
        file_records.append({"name": filename, "size": len(content)})

    filenames = [r["name"] for r in file_records]
    bad = validate_extensions(filenames)
    if bad:
        raise HTTPException(400, f"Unsupported file extension(s): {', '.join(bad)}")

    total_bytes = sum(r["size"] for r in file_records)
    if total_bytes > 2 * 1024 ** 3:
        raise HTTPException(413, "Payload too large (limit 2 GB).")

    upload_id  = new_id("upl")
    input_type = detect_input_type(filenames)
    ok_msg     = (
        "Valid DICOM series detected."
        if input_type == "dicom"
        else "Valid NIfTI input detected."
    )

    record = {
        "upload_id":           upload_id,
        "detected_input_type": input_type,
        "file_count":          len(file_records),
        "files":               file_records,
        "validation":          {"ok": True, "message": ok_msg},
    }
    uploads[upload_id] = record
    return record

# ---------------------------------------------------------------------------
# Endpoint 2: Explicit validation
# ---------------------------------------------------------------------------

@app.post("/v1/validate-input")
def validate_input(body: ValidateInputRequest):
    upload = uploads.get(body.upload_id)
    if not upload:
        raise HTTPException(404, "upload_id not found.")

    input_type = upload["detected_input_type"]
    message = (
        "Valid NIfTI input detected. Ready to continue."
        if input_type == "nifti"
        else "Valid DICOM series detected. Ready to continue."
    )
    return {"ok": True, "detected_input_type": input_type, "message": message}

# ---------------------------------------------------------------------------
# Endpoint 3: Create pipeline job
# ---------------------------------------------------------------------------

@app.post("/v1/jobs", status_code=202)
async def create_job(body: CreateJobRequest):
    if body.upload_id not in uploads:
        raise HTTPException(404, "upload_id not found.")

    job_id = new_id("job")
    job = {
        "job_id":             job_id,
        "upload_id":          body.upload_id,
        "pipeline_name":      body.pipeline_name,
        "has_external_chest": body.has_external_chest,
        "report_format":      body.report_format,
        "notes":              body.notes,
        "status":             "queued",
        "progress":           0,
        "steps":              initial_steps(),
        "message":            "Job queued.",
        "created_at":         now_iso(),
        "updated_at":         now_iso(),
        "result":             None,
    }
    jobs[job_id] = job
    asyncio.create_task(run_pipeline(job_id))
    return {"job_id": job_id, "status": "queued", "created_at": job["created_at"]}

# ---------------------------------------------------------------------------
# Endpoint 4: Job status + progress
# ---------------------------------------------------------------------------

@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return {
        "job_id":     job["job_id"],
        "status":     job["status"],
        "progress":   job["progress"],
        "steps":      job["steps"],
        "message":    job["message"],
        "updated_at": job["updated_at"],
    }

# ---------------------------------------------------------------------------
# Endpoint 5: Final result
# ---------------------------------------------------------------------------

@app.get("/v1/jobs/{job_id}/result")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "completed":
        raise HTTPException(409, f"Job not finished yet (status: {job['status']}).")
    return job["result"]

# ---------------------------------------------------------------------------
# Endpoint 6: Download report
# ---------------------------------------------------------------------------

@app.get("/v1/jobs/{job_id}/report")
def download_report(
    job_id: str,
    format: Literal["csv", "json"] = Query("csv"),
):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "completed":
        raise HTTPException(409, "Job not finished yet.")

    row = job["result"]["report_row"]

    if format == "json":
        data     = json.dumps(row, indent=2).encode()
        filename = "pipeline_report.json"
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    buf    = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(row.keys()))
    writer.writeheader()
    writer.writerow(row)
    data = buf.getvalue().encode()
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="pipeline_report.csv"'},
    )

# ---------------------------------------------------------------------------
# Endpoint 7: Download artifact
# ---------------------------------------------------------------------------

@app.get("/v1/jobs/{job_id}/artifacts/{artifact_name}")
def download_artifact(job_id: str, artifact_name: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "completed":
        raise HTTPException(409, "Job not finished yet.")

    # Swap this out for real file streaming in production.
    placeholder = (
        f"# Placeholder artifact: {artifact_name}\n"
        f"# Job: {job_id}\n"
        f"# Generated: {now_iso()}\n"
    ).encode()

    media_type = "application/octet-stream"
    if artifact_name.endswith((".nii.gz", ".nii")):
        media_type = "application/gzip"
    elif artifact_name.endswith(".dcm"):
        media_type = "application/dicom"

    return StreamingResponse(
        io.BytesIO(placeholder),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{artifact_name}"'},
    )

# ---------------------------------------------------------------------------
# Endpoint 8: Cancel job
# ---------------------------------------------------------------------------

@app.delete("/v1/jobs/{job_id}", status_code=202)
def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(409, f"Job already in terminal state: {job['status']}.")

    job["status"]     = "cancelled"
    job["message"]    = "Job cancelled by user."
    job["updated_at"] = now_iso()
    return {"job_id": job_id, "status": "cancelled"}
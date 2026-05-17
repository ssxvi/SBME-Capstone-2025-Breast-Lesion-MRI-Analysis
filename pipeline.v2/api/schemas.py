"""
api/schemas.py
--------------
Pydantic models for the FastAPI request and response layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PipelineStatus(str, Enum):
    pending   = "pending"
    running   = "running"
    complete  = "complete"
    failed    = "failed"


class LesionLabel(str, Enum):
    no_lesion = "no_lesion"
    lesion    = "lesion"


class MalignancyLabel(str, Enum):
    benign    = "benign"
    malignant = "malignant"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class LesionResult(BaseModel):
    label:         LesionLabel
    confidence:    float = Field(..., ge=0.0, le=1.0)
    probabilities: list[float] = Field(..., min_length=2, max_length=2)


class SegmentationResult(BaseModel):
    lesion_volume_voxels: int
    lesion_area_pixels:   Optional[int] = None
    mask_path:            str
    preview_image_url:    Optional[str] = None


class MalignancyResult(BaseModel):
    label:         MalignancyLabel
    confidence:    float = Field(..., ge=0.0, le=1.0)
    probabilities: list[float] = Field(..., min_length=2, max_length=2)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class RunPipelineRequest(BaseModel):
    """
    Submitted by the frontend to start a pipeline run.
    File paths are resolved server-side after upload.
    """
    case_id:               str
    pre_path:              str = Field(..., description="Server path to Pre timepoint .nii.gz")
    post1_path:            str = Field(..., description="Server path to Post_1 timepoint .nii.gz")
    post2_path:            str = Field(..., description="Server path to Post_2 timepoint .nii.gz")
    skip_segmentation:     bool  = False
    use_mask_for_roi:      bool  = True
    lesion_threshold:      float = Field(0.5, ge=0.0, le=1.0)
    nnunet_dataset_id:     str   = "001"
    nnunet_configuration:  str   = "3d_fullres"
    nnunet_fold:           str   = "0"
    demo_mode:             bool  = Field(False, description="Simulate a run for demos without executing heavy models")
    demo_duration_sec:     float = Field(10.0, ge=1.0, le=300.0, description="Approximate simulated run duration when demo_mode is enabled")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PipelineRunResponse(BaseModel):
    """Returned immediately when a pipeline job is accepted."""
    run_id:  str
    case_id: str
    status:  PipelineStatus = PipelineStatus.pending
    demo_mode: bool = False


class PipelineResultResponse(BaseModel):
    """Full result, polled after the run completes."""
    run_id:     str
    case_id:    str
    status:     PipelineStatus
    error:      Optional[str] = None
    current_stage: Optional[str] = None
    progress:      Optional[float] = Field(None, ge=0.0, le=1.0)

    lesion:       Optional[LesionResult]       = None
    segmentation: Optional[SegmentationResult] = None
    malignancy:   Optional[MalignancyResult]   = None

    overall_impression: Optional[str] = None
    report_url:         Optional[str] = None   # e.g. /report/{run_id}


class UploadResponse(BaseModel):
    """Returned after a successful file upload."""
    filename:    str
    server_path: str
    size_bytes:  int


class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str = "1.0.0"

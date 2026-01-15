"""
KizunaSR API - Pydantic Schemas
================================
Request/Response models for API endpoints.
"""

from pydantic import BaseModel
from typing import Literal, Optional
from enum import Enum


class JobState(str, Enum):
    """Job processing states."""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessMode(str, Enum):
    """Processing modes."""
    IMAGE = "image"
    VIDEO = "video"


# === Response Models ===

class UploadResponse(BaseModel):
    """Response after successful file upload."""
    job_id: str
    filename: str


class ProcessRequest(BaseModel):
    """Request to start processing a job."""
    job_id: str
    mode: ProcessMode
    scale: Literal[2, 4] = 2


class StatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    state: JobState
    progress: int = 0
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    detail: Optional[str] = None

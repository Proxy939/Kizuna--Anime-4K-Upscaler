from pydantic import BaseModel
from typing import Optional, Literal
from enum import Enum

class JobState(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class UploadResponse(BaseModel):
    job_id: str
    filename: str

class ProcessRequest(BaseModel):
    job_id: str
    mode: Literal["image", "video"]
    scale: int = 2
    model: Optional[str] = None
    enhance: bool = False  # Optional Anime4K post-processing

class StatusResponse(BaseModel):
    state: JobState
    progress: int

class ErrorResponse(BaseModel):
    detail: str

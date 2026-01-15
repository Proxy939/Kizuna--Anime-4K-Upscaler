"""
KizunaSR API - Job Manager
===========================
In-memory job state tracker with filesystem storage paths.
"""

import uuid
from typing import Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from .schemas import JobState


@dataclass
class Job:
    """Represents a processing job."""
    job_id: str
    filename: str
    input_path: Path
    output_path: Optional[Path] = None
    state: JobState = JobState.UPLOADED
    progress: int = 0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    mode: Optional[str] = None
    scale: int = 2


class JobManager:
    """
    In-memory job state tracker.
    
    Manages job lifecycle: uploaded → queued → processing → completed | failed
    """
    
    def __init__(self, uploads_dir: Path, outputs_dir: Path):
        self.uploads_dir = uploads_dir
        self.outputs_dir = outputs_dir
        self.jobs: Dict[str, Job] = {}
        
        # Ensure directories exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
    
    def create_job(self, filename: str, input_path: Path) -> Job:
        """Create a new job after file upload."""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            filename=filename,
            input_path=input_path,
            state=JobState.UPLOADED
        )
        self.jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def update_state(self, job_id: str, state: JobState, 
                     progress: int = None, error: str = None) -> Optional[Job]:
        """Update job state and progress."""
        job = self.jobs.get(job_id)
        if job:
            job.state = state
            if progress is not None:
                job.progress = progress
            if error is not None:
                job.error = error
        return job
    
    def set_output(self, job_id: str, output_path: Path) -> Optional[Job]:
        """Set the output file path for a completed job."""
        job = self.jobs.get(job_id)
        if job:
            job.output_path = output_path
        return job
    
    def queue_job(self, job_id: str, mode: str, scale: int) -> Optional[Job]:
        """Queue a job for processing."""
        job = self.jobs.get(job_id)
        if job:
            job.state = JobState.QUEUED
            job.mode = mode
            job.scale = scale
        return job

"""
KizunaSR API Server
====================
FastAPI adapter layer that wraps the existing KizunaSR backend library.

This module provides HTTP endpoints to:
1. Upload files
2. Trigger processing (calls existing backend pipeline)
3. Poll job status
4. Download processed results

IMPORTANT: This is an adapter layer only. It does NOT modify any backend logic.
The backend in core/, runtime/, tools/ remains completely untouched.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Add project root to path for backend imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .schemas import (
    UploadResponse, ProcessRequest, StatusResponse, 
    ErrorResponse, JobState, ProcessMode
)
from .job_manager import JobManager

# === Configuration ===
API_DIR = Path(__file__).parent
UPLOADS_DIR = API_DIR / "storage" / "uploads"
OUTPUTS_DIR = API_DIR / "storage" / "outputs"

MAX_FILE_SIZE = 512 * 1024 * 1024  # 512MB
ALLOWED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4"
}

# === Initialize App ===
app = FastAPI(
    title="KizunaSR API",
    description="API adapter for KizunaSR anime upscaling pipeline",
    version="1.0.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize job manager
job_manager = JobManager(UPLOADS_DIR, OUTPUTS_DIR)


# === Helper Functions ===

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file type and size."""
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: JPEG, PNG, WEBP, MP4"
        )


async def process_job_background(job_id: str) -> None:
    """
    Background task to process a job using the existing backend pipeline.
    
    This function wraps the existing KizunaSR pipeline without modifying it.
    """
    job = job_manager.get_job(job_id)
    if not job:
        return
    
    try:
        # Update state to processing
        job_manager.update_state(job_id, JobState.PROCESSING, progress=10)
        
        # Import backend pipeline (lazy import to avoid startup delays)
        from core.pipeline import KizunaSRPipeline, PipelineConfig
        from PIL import Image
        
        # Determine processing mode
        is_video = job.mode == ProcessMode.VIDEO.value
        
        if is_video:
            # Video processing
            job_manager.update_state(job_id, JobState.PROCESSING, progress=20)
            
            from runtime.video.video_core import VideoProcessor
            
            output_filename = f"{job.job_id}_upscaled.mp4"
            output_path = OUTPUTS_DIR / output_filename
            
            # Configure pipeline
            config = PipelineConfig()
            config.use_ai = True
            config.scale_factor = job.scale
            
            # Process video
            processor = VideoProcessor(config)
            job_manager.update_state(job_id, JobState.PROCESSING, progress=30)
            
            processor.process_video(
                str(job.input_path),
                str(output_path)
            )
            
            job_manager.update_state(job_id, JobState.PROCESSING, progress=90)
            
        else:
            # Image processing
            job_manager.update_state(job_id, JobState.PROCESSING, progress=20)
            
            # Load input image
            input_image = Image.open(job.input_path).convert("RGB")
            
            job_manager.update_state(job_id, JobState.PROCESSING, progress=40)
            
            # Configure pipeline
            config = PipelineConfig()
            config.use_ai = True
            config.scale_factor = job.scale
            
            # Create and run pipeline
            pipeline = KizunaSRPipeline(config)
            result_image = pipeline.process_frame(input_image)
            
            job_manager.update_state(job_id, JobState.PROCESSING, progress=80)
            
            # Save output
            output_filename = f"{job.job_id}_upscaled.png"
            output_path = OUTPUTS_DIR / output_filename
            result_image.save(output_path, "PNG")
            
            job_manager.update_state(job_id, JobState.PROCESSING, progress=90)
        
        # Mark as completed
        job_manager.set_output(job_id, output_path)
        job_manager.update_state(job_id, JobState.COMPLETED, progress=100)
        
    except Exception as e:
        # Mark as failed with error message
        job_manager.update_state(
            job_id, 
            JobState.FAILED, 
            error=str(e)
        )


# === API Endpoints ===

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for processing.
    
    Accepts: JPEG, PNG, WEBP images or MP4 video
    Max size: 512MB
    
    Returns job_id for tracking.
    """
    # Validate file
    validate_file(file)
    
    # Generate unique filename
    import uuid
    ext = ALLOWED_MIME_TYPES.get(file.content_type, ".bin")
    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}{ext}"
    file_path = UPLOADS_DIR / safe_filename
    
    # Ensure uploads directory exists
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save file with size check
    total_size = 0
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                buffer.close()
                file_path.unlink()  # Delete partial file
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: 512MB"
                )
            buffer.write(chunk)
    
    # Create job
    job = job_manager.create_job(file.filename, file_path)
    
    return UploadResponse(
        job_id=job.job_id,
        filename=file.filename
    )


@app.post("/api/process", response_model=StatusResponse)
async def process_file(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Start processing an uploaded file.
    
    The processing runs in the background (non-blocking).
    Poll /api/status/{job_id} to check progress.
    """
    job = job_manager.get_job(request.job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.state not in [JobState.UPLOADED, JobState.FAILED]:
        raise HTTPException(
            status_code=400, 
            detail=f"Job cannot be processed in state: {job.state.value}"
        )
    
    # Queue the job
    job_manager.queue_job(request.job_id, request.mode.value, request.scale)
    
    # Start background processing
    background_tasks.add_task(process_job_background, request.job_id)
    
    return StatusResponse(
        job_id=job.job_id,
        state=JobState.QUEUED,
        progress=0
    )


@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """
    Get the current status of a job.
    
    States: uploaded, queued, processing, completed, failed
    Progress: 0-100
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return StatusResponse(
        job_id=job.job_id,
        state=job.state,
        progress=job.progress,
        error=job.error
    )


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """
    Download the processed result file.
    
    Only available after job state is 'completed'.
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.state != JobState.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Result not ready. Current state: {job.state.value}"
        )
    
    if not job.output_path or not job.output_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    # Determine media type
    suffix = job.output_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".webp": "image/webp",
        ".mp4": "video/mp4"
    }
    media_type = media_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        path=job.output_path,
        filename=f"kizuna_upscaled_{job.filename}",
        media_type=media_type
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "KizunaSR API"}


# === Run Server ===

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("KizunaSR API Server")
    print("=" * 60)
    print(f"Uploads directory: {UPLOADS_DIR}")
    print(f"Outputs directory: {OUTPUTS_DIR}")
    print("=" * 60)
    
    uvicorn.run(
        "api.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

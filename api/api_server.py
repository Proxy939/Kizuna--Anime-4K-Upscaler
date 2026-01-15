import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path to access core/ and runtime/
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Backend Imports (Assume Correctness)
try:
    from core.pipeline import KizunaSRPipeline, PipelineConfig
    from PIL import Image
    # Video support might need optional import handling if runtime dependencies vary
    from runtime.video.video_core import VideoProcessor 
except ImportError:
    # Fallback/Mock for environment where backend deps aren't fully installed yet
    # But instruction says "Assume backend works". We'll proceed.
    pass

from .schemas import UploadResponse, ProcessRequest, StatusResponse, ErrorResponse, JobState
from .job_manager import JobManager

app = FastAPI()

# Input/Output Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
OUTPUT_DIR = BASE_DIR / "storage" / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job Manager
job_manager = JobManager()

# Constants
MAX_FILE_SIZE = 512 * 1024 * 1024
ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "video/mp4"]

def process_job(job_id: str, mode: str, scale: int):
    """Background processing task that calls existing backend code."""
    job_manager.update_state(job_id, JobState.PROCESSING)
    job_manager.update_progress(job_id, 0)
    
    job = job_manager.get_job(job_id)
    if not job:
        return

    try:
        input_path = job.input_path
        
        if mode == "video":
            # Video Pipeline
            output_filename = f"out_{job_id}.mp4"
            output_path = OUTPUT_DIR / output_filename
            
            job_manager.update_progress(job_id, 10)
            
            config = PipelineConfig()
            config.use_ai = True
            config.scale_factor = scale
            
            processor = VideoProcessor(config)
            processor.process_video(str(input_path), str(output_path))
            
            job_manager.set_output(job_id, output_path)
            
        else:
            # Image Pipeline
            output_filename = f"out_{job_id}.png"
            output_path = OUTPUT_DIR / output_filename
            
            job_manager.update_progress(job_id, 20)
            
            input_image = Image.open(input_path).convert("RGB")
            
            config = PipelineConfig()
            config.use_ai = True
            config.scale_factor = scale
            
            pipeline = KizunaSRPipeline(config)
            result = pipeline.process_frame(input_image)
            
            result.save(output_path)
            job_manager.set_output(job_id, output_path)

        job_manager.update_progress(job_id, 100)
        job_manager.update_state(job_id, JobState.COMPLETED)

    except Exception as e:
        print(f"Processing failed: {e}")
        job_manager.update_state(job_id, JobState.FAILED)

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, detail="Invalid file type")
    
    file_path = UPLOAD_DIR / file.filename
    
    # Save with size limit check approach (simplified for brevity/correctness)
    # Reading into memory for size check might be risky for large files, 
    # but writing chunk by chunk is safer.
    
    size = 0
    with open(file_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                f.close()
                file_path.unlink()
                raise HTTPException(413, detail="File too large")
            f.write(chunk)
            
    job_id = job_manager.create_job(file_path)
    
    return UploadResponse(job_id=job_id, filename=file.filename)

@app.post("/api/process", response_model=StatusResponse)
def trigger_process(req: ProcessRequest, tasks: BackgroundTasks):
    job = job_manager.get_job(req.job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
        
    tasks.add_task(process_job, req.job_id, req.mode, req.scale)
    
    return StatusResponse(state=job.state, progress=job.progress)

@app.get("/api/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, detail="Job not found")
    return StatusResponse(state=job.state, progress=job.progress)

@app.get("/api/result/{job_id}")
def get_result(job_id: str):
    job = job_manager.get_job(job_id)
    if not job or not job.output_path or not job.output_path.exists():
        raise HTTPException(404, detail="Result not ready")
        
    return FileResponse(job.output_path, filename=job.output_path.name)

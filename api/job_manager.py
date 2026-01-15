import uuid
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from .schemas import JobState

@dataclass
class Job:
    job_id: str
    input_path: Path
    output_path: Optional[Path] = None
    state: JobState = JobState.UPLOADED
    progress: int = 0

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def create_job(self, input_path: Path) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = Job(job_id=job_id, input_path=input_path)
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def update_progress(self, job_id: str, progress: int):
        if job_id in self.jobs:
            self.jobs[job_id].progress = progress

    def update_state(self, job_id: str, state: JobState):
        if job_id in self.jobs:
            self.jobs[job_id].state = state

    def set_output(self, job_id: str, output_path: Path):
        if job_id in self.jobs:
            self.jobs[job_id].output_path = output_path

# KizunaSR API

FastAPI adapter layer that wraps the existing KizunaSR backend library.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn python-multipart

# Run the server
cd "d:\college\Projects\Kizuna (絆)"
python -m api.api_server
```

Server runs at: http://localhost:8000

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload file, returns job_id |
| `/api/process` | POST | Start processing |
| `/api/status/{job_id}` | GET | Poll job status |
| `/api/result/{job_id}` | GET | Download result |
| `/api/health` | GET | Health check |

## Example Usage

### Upload
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/api/upload
```

### Process
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"job_id": "uuid", "mode": "image", "scale": 2}' \
  http://localhost:8000/api/process
```

### Check Status
```bash
curl http://localhost:8000/api/status/{job_id}
```

### Download Result
```bash
curl -O http://localhost:8000/api/result/{job_id}
```

## File Limits

- Max size: 512MB
- Allowed types: JPEG, PNG, WEBP, MP4

## Architecture

This is an **adapter layer only**. It does NOT modify any backend code.

```
api/
├── api_server.py     # FastAPI endpoints
├── job_manager.py    # Job state tracker
├── schemas.py        # Request/response models
└── storage/
    ├── uploads/      # Uploaded files
    └── outputs/      # Processed results
```

The backend in `core/`, `runtime/`, `tools/` remains untouched.

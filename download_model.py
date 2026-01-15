"""Download Real-ESRGAN model file from GitHub releases."""
import requests
from pathlib import Path

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-animevideov3.pth"
OUTPUT_PATH = Path("models/realesrgan-animevideov3.pth")

print(f"Downloading from: {MODEL_URL}")
print(f"Saving to: {OUTPUT_PATH}")

try:
    # Stream download with progress
    response = requests.get(MODEL_URL, stream=True, allow_redirects=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    print(f"File size: {total_size / 1024 / 1024:.1f} MB")
    
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    
    with open(OUTPUT_PATH, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    
    print(f"\n✅ Download complete: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    
except requests.exceptions.RequestException as e:
    print(f"\n❌ Download failed: {e}")
    print("\nPlease download manually:")
    print("1. Visit: https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0")
    print("2. Download: realesrgan-animevideov3.pth")
    print(f"3. Save to: {OUTPUT_PATH.absolute()}")

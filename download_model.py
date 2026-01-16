import os
import requests
from tqdm import tqdm

# Updated to v0.2.5.0 and RealESRGAN_x4plus_anime_6B (saved as standard name)
URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus_anime_6B.pth"
OUT_PATH = os.path.join("models", "RealESRGAN_x4plus.pth")

os.makedirs("models", exist_ok=True)

print(f"Downloading Real-ESRGAN x4+ Anime (Default ⭐)...")
print(f"Source: {URL}")
print(f"Target: {OUT_PATH}")

try:
    response = requests.get(URL, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB

    with open(OUT_PATH, "wb") as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print("\n✅ Download complete:", OUT_PATH)

except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\n⚠️  Manual Download Required:")
    print(f"1. Download: {URL}")
    print(f"2. Save as: {os.path.abspath(OUT_PATH)}")

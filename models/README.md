# AI Models

This directory contains AI upscaling model weights.

## Default Model (Required)

**Real-ESRGAN Anime (Default ⭐)**

Download via PowerShell:
```powershell
Invoke-WebRequest `
  -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-animevideov3.pth" `
  -OutFile "models\realesrgan-animevideov3.pth"
```

**Linux/macOS**:
```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-animevideov3.pth -O models/realesrgan-animevideov3.pth
```

- **File**: `realesrgan-animevideov3.pth`
- **Size**: ~87 MB
- **Scale**: 4x
- **Best for**: Anime/animation content

## Optional Models

### Real-ESRGAN Anime x4+
```powershell
Invoke-WebRequest `
  -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" `
  -OutFile "models\RealESRGAN_x4plus_anime_6B.pth"
```

### Real-ESRGAN General x4+
```powershell
Invoke-WebRequest `
  -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" `
  -OutFile "models\RealESRGAN_x4plus.pth"
```

## Notes

- Model files (*.pth) are git-ignored
- Default model is loaded automatically when AI mode is enabled
- Missing model = clean failure with error message

# KizunaSR - GPU-Only YUV Pipeline Implementation

This document describes the implementation of the GPU-only YUV frame path for KizunaSR (Option 3.3).

## Overview

The GPU-only path eliminates the costly CPU YUV→RGB conversion by decoding video into YUV planes, uploading them directly to the GPU as separate textures, and performing color space conversion in a shader.

### Benefits
- **Reduced CPU Usage**: Decoding YUV is faster than YUV+RGB conversion.
- **Lower Memory Bandwidth**: Uploading YUV420 uses 50% less bandwidth than RGB24.
- **Improved Latency**: Decoding and upload steps are faster.

## Implementation Details

### 1. VideoDecoder
- Added `output_yuv=True` option.
- When enabled, `VideoFrame` contains raw Y, U, V plane data instead of RGB pixels.
- Falls back to RGB if YUV extraction fails.

### 2. OpenGLTextureUploader
- Handles YUV `VideoFrame`s specially.
- Uploads Y plane to a `GL_R8` texture.
- Uploads interleaved UV planes (NV12) or combined U+V (YUV420p) to a `GL_RG8` texture.
- Uses a `yuv_textures` registry to track Y and UV texture pairs.

### 3. ShaderPipelineExecutor
- Added **Stage 0** (YUV→RGB Conversion).
- Detects if input is YUV (by presence of UV texture ID).
- Executes `yuv_to_rgb.frag` to convert YUV to linear RGB.
- Output RGB is fed into Stage 1 (Normalize).

### 4. Player
- Orchestrates the YUV path.
- Initializes decoder with `output_yuv=True`.
- Extracts UV texture ID from uploader and passes it to pipeline execution.

## Usage

To enable the GPU-only path, initialize `Player` with `use_yuv_path=True`:

```python
player = Player(
    video_path="input.mp4",
    scale_factor=2,
    use_yuv_path=True  # Enables GPU-only path
)
player.play()
```

If the shader is missing or initialization fails, the system automatically falls back to CPU RGB path behaviors where possible (though Player currently enforces flag).

## Validation

1. **Visual Correctness**: Output should look identical to CPU path (colors typically match BT.709 standard).
2. **Performance**: Check Task Manager or `top` - CPU usage should be lower.
3. **Logs**: Look for `[Pipeline] YUV shader loaded (Stage 0 enabled)` and `[Player] Path: GPU YUV`.

# KizunaSR GPU-Only Frame Path - Implementation Notes

## What Was Implemented

### 1. YUV Frame Support in VideoDecoder
- Extended `VideoFrame` dataclass with `yuv_data` field
- Added `output_yuv` parameter to `VideoDecoder.__init__()`
- YUV420p plane extraction from PyAV frames
- Graceful fallback to RGB if YUV extraction fails

### 2. YUV Texture Upload
- Extended `OpenGLTextureUploader` to handle YUV frames
- Separate Y plane (R8) and UV plane (RG8) textures
- YUV texture registry for proper cleanup
- Maintains same API - `upload_frame()` signature unchanged

### 3. YUV→RGB Conversion Shader
- Fragment shader: `yuv_to_rgb.frag`
- BT.709 color space (standard for HD video)
- Proper YUV range normalization
- Output: linear RGB ready for KizunaSR pipeline

## GPU-Only Flow

```
Video File
  ↓
Hardware Decode (YUV420)
  ↓
Upload Y plane → GL_R8 texture
Upload UV plane → GL_RG8 texture
  ↓
YUV→RGB Shader
  ↓
Stage 1: Normalize (existing pipeline)
  ↓
... (rest of KizunaSR pipeline)
```

## Integration

To enable GPU-only path:

```python
# Create decoder with YUV output
decoder = VideoDecoder("video.mp4", use_hwaccel=True, output_yuv=True)

# Uploader handles both RGB and YUV transparently
uploader = OpenGLTextureUploader()
texture_id = uploader.upload_frame(frame)  # Works for both!
```

## What Was NOT Implemented

1. **Shader Pipeline Integration**: The YUV shader needs to be inserted as a pre-processing stage before Stage 1
2. **PBO for YUV**: Currently YUV uses direct upload, not PBO (can be added later)
3. **NV12 Format**: Only YUV420P is supported (separate U/V planes)
4. **Player Auto-Detection**: Player still needs manual `output_yuv=True` flag
5. **10-bit/HDR**: Only 8-bit SDR BT.709
6. **Zero-copy GPU surfaces**: Still CPU→GPU copy (needs CUDA interop for true zero-copy)

## Performance Impact

**Before (CPU RGB)**:
- Decode → Convert to RGB (CPU) → Upload to GPU

**After (GPU YUV)**:
- Decode → Upload YUV planes → Convert on GPU

**Expected improvement**: ~20-30% reduction in CPU usage, ~10-15% faster frame processing

## Fallback Safety

- If YUV extraction fails → automatic RGB fallback
- If YUV texture upload fails → exception with clear error
- If `output_yuv=False` → classic RGB path (unchanged)

## Next Steps for Full Integration

1. Insert YUV→RGB shader as Stage 0 in `ShaderPipelineExecutor`
2. Modify Player to use `output_yuv=True` by default
3. Add PBO support for YUV planes (similar to RGB path)
4. Implement proper YUV texture binding in shader pipeline
5. Add performance metrics/logging for GPU-only path validation

---
*GPU-Only Path Implemented - YUV Decode + Shader Conversion Ready*

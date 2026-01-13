# KizunaSR Shared Video Core

## Overview
Mode-agnostic video decoding and frame management system using FFmpeg (via PyAV). Provides frame extraction, timing preservation, and an abstract interface for GPU texture upload.

**Purpose**: Foundation for both offline video processing and real-time playback.

## Architecture

### Components

1. **VideoFrame**: Data structure representing a decoded frame
2. **VideoDecoder**: FFmpeg-based video decoder (PyAV)
3. **FrameQueue**: Thread-safe frame buffer
4. **IGPUTextureUploader**: Abstract interface for GPU upload
5. **VideoFrameProcessor**: Orchestrates frame lifecycle

### Data Flow

```
Video File
  ↓ VideoDecoder (FFmpeg/PyAV)
  ↓ VideoFrame (raw RGB data + metadata)
  ↓ FrameQueue (buffering)
  ↓ IGPUTextureUploader (abstract upload)
  ↓ GPU Texture (graphics API-specific)
  ↓ Downstream processing (shader/display/encode)
```

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies**:
- `av >= 10.0.0` (PyAV: FFmpeg bindings)
- `numpy >= 1.24.0`

## Usage

### Basic Video Decoding

```python
from runtime.video import VideoDecoder

# Open video
with VideoDecoder("input.mp4") as decoder:
    print(f"Resolution: {decoder.width}×{decoder.height}")
    print(f"FPS: {decoder.fps}")
    print(f"Codec: {decoder.codec}")
    
    # Iterate through frames
    for frame in decoder:
        print(f"Frame {frame.frame_index}: {frame.width}×{frame.height}, PTS={frame.pts:.3f}s")
        # frame.data is numpy array (H, W, 3) RGB uint8
```

### Frame Queue Management

```python
from runtime.video import VideoDecoder, FrameQueue

decoder = VideoDecoder("input.mp4")
queue = FrameQueue(maxsize=16)

# Producer: decode and enqueue
for frame in decoder:
    queue.enqueue(frame)
    
    # Consumer: dequeue and process
    queued_frame = queue.dequeue()
    if queued_frame:
        # Process frame
        pass

# Check stats
stats = queue.stats()
print(f"Processed {stats['total_dequeued']} frames")
```

### Complete Frame Processing

```python
from runtime.video import VideoFrameProcessor

def process_frame(frame, texture_id):
    """Callback called for each frame."""
    print(f"Processing frame {frame.frame_index}")
    # texture_id is None if no GPU uploader
    # frame.data contains RGB pixel data

processor = VideoFrameProcessor("input.mp4", gpu_uploader=None)
processor.process_all_frames(process_frame)
processor.close()
```

### GPU Upload Interface (Abstract)

```python
from runtime.video import IGPUTextureUploader, VideoFrame

class MyGPUUploader(IGPUTextureUploader):
    def upload_frame(self, frame: VideoFrame) -> int:
        # Implement OpenGL/Vulkan/WebGPU texture upload
        # Example (OpenGL):
        # texture_id = glGenTextures(1)
        # glBindTexture(GL_TEXTURE_2D, texture_id)
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.width, frame.height,
        #              0, GL_RGB, GL_UNSIGNED_BYTE, frame.data)
        return texture_id
    
    def release_texture(self, texture_id: int):
        # Implement texture deletion
        # glDeleteTextures([texture_id])
        pass
    
    def get_texture_size(self, texture_id: int) -> tuple[int, int]:
        # Return texture dimensions
        return (width, height)

# Use with processor
uploader = MyGPUUploader()
processor = VideoFrameProcessor("input.mp4", gpu_uploader=uploader)
processor.process_all_frames(my_callback)
```

## Data Structures

### VideoFrame

```python
@dataclass
class VideoFrame:
    data: np.ndarray      # (H, W, 3) RGB uint8
    width: int            # Frame width
    height: int           # Frame height
    pts: float            # Presentation timestamp (seconds)
    frame_index: int      # Sequential frame number
    format: str           # Color format ('rgb24')
```

**Ownership**: Caller owns the `VideoFrame` object after dequeuing. No automatic cleanup.

**Lifetime**: Valid until overwritten or explicitly deleted. Copy `frame.data` if needed beyond immediate processing.

### Supported Formats

**Containers**: MP4, MKV, AVI, WebM (anything FFmpeg supports)  
**Codecs**: H.264, H.265/HEVC, VP8, VP9, AV1, etc.  
**Output format**: Always RGB24 (8-bit per channel)

## Frame Timing

### PTS Preservation

**Presentation Timestamp (PTS)**:
- Stored in `VideoFrame.pts` (float, seconds)
- Calculated from stream time base
- Preserves original video timing

**Variable Frame Rate (VFR)**:
- Full VFR support
- PTS used for accurate timing, not assumed constant FPS

**Frame Index**:
- Sequential counter (0-indexed)
- Independent of PTS
- Useful for frame-by-frame processing

### Seeking

```python
decoder = VideoDecoder("input.mp4")

# Seek to 10.5 seconds
decoder.seek(10.5)

# Continue decoding from new position
for frame in decoder:
    print(f"Frame from {frame.pts:.3f}s")
```

**Note**: Seeking accuracy depends on video keyframes (I-frames).

## Frame Lifecycle

### States

```
1. DECODED    → VideoDecoder yields VideoFrame
2. ENQUEUED   → FrameQueue.enqueue()
3. PROCESSING → FrameQueue.dequeue()
4. UPLOADED   → IGPUTextureUploader.upload_frame() (optional)
5. RELEASED   → IGPUTextureUploader.release_texture() (optional)
```

### Memory Management

**CPU Memory**:
- `VideoFrame.data` is a numpy array
- Released when VideoFrame goes out of scope
- Queue size limits total RAM usage

**GPU Memory** (if using uploader):
- Managed by IGPUTextureUploader implementation
- **CRITICAL**: Must call `release_texture()` to avoid leaks
- Uploader implementation responsible for tracking textures

### Queue Bounds

Default queue size: 16 frames

**Calculate memory usage**:
```
RAM per frame = width × height × 3 bytes
Total RAM = queue_size × RAM_per_frame

Example (1080p):
1920 × 1080 × 3 = 6.2 MB/frame
16 frames = ~100 MB
```

## Error Handling

### Decoder Errors

```python
try:
    decoder = VideoDecoder("video.mp4")
except FileNotFoundError:
    print("Video file not found")
except ValueError as e:
    print(f"Cannot open video: {e}")
```

**Common errors**:
- File not found
- Unsupported codec
- Corrupted file
- No video stream

### Frame Processing Errors

```python
for frame in decoder:
    try:
        # Process frame
        result = process(frame)
    except Exception as e:
        print(f"Error processing frame {frame.frame_index}: {e}")
        continue  # Skip frame, continue processing
```

**Graceful degradation**: Skip bad frames, log errors, continue processing.

## Validation

### Frame Order Validation

```python
decoder = VideoDecoder("video.mp4")
prev_index = -1
prev_pts = -1.0

for frame in decoder:
    assert frame.frame_index == prev_index + 1, "Frame index not sequential"
    assert frame.pts >= prev_pts, "PTS not monotonically increasing"
    
    prev_index = frame.frame_index
    prev_pts = frame.pts
```

### Frame Loss Detection

```python
decoder = VideoDecoder("video.mp4")
expected_frames = decoder.total_frames

frame_count = 0
for frame in decoder:
    frame_count += 1

if expected_frames:
    assert frame_count == expected_frames, f"Frame loss: {expected_frames - frame_count} missing"
```

### Resolution & Format Validation

```python
decoder = VideoDecoder("video.mp4")

for frame in decoder:
    assert frame.width == decoder.width, "Width mismatch"
    assert frame.height == decoder.height, "Height mismatch"
    assert frame.format == 'rgb24', "Format not RGB24"
    assert frame.data.shape == (decoder.height, decoder.width, 3), "Data shape mismatch"
```

## Performance Considerations

### Decoding Speed

**Single-threaded decoding**: ~100-300 fps (depends on codec/resolution)

**Example (RTX 3060, H.264 1080p)**:
- Decode: ~200 fps (~5ms/frame)
- RGB conversion: ~50 fps (~20ms/frame)
- **Total**: ~40-50 fps

**Bottleneck**: RGB conversion (color space transform)

### Optimization Opportunities (Future)

1. **Hardware decoding**: NVDEC, VAAPI, QSV (via PyAV)
2. **Multi-threaded decoding**: Process frames in parallel
3. **Zero-copy GPU upload**: Direct YUV→RGB on GPU

## Integration Examples

### With KizunaSR Pipeline

```python
from runtime.video import VideoFrameProcessor
from core.pipeline import KizunaSRPipeline, PipelineConfig
from PIL import Image

config = PipelineConfig()
config.use_ai = True
config.ai_model_path = "models/anime_sr_2x.onnx"

pipeline = KizunaSRPipeline(config)

def process_frame(frame, texture_id):
    # Convert VideoFrame to PIL Image
    input_img = Image.fromarray(frame.data, mode='RGB')
    
    # Run KizunaSR pipeline
    output_img = pipeline.process_frame(input_img)
    
    # Save or encode output
    output_img.save(f"output_frame_{frame.frame_index:06d}.png")

processor = VideoFrameProcessor("input.mp4")
processor.process_all_frames(process_frame)
```

### Real-Time Playback (Future)

```python
from runtime.video import VideoFrameProcessor
import time

def realtime_callback(frame, texture_id):
    # Upload to GPU (texture_id provided if uploader used)
    # Bind texture to shader pipeline
    # Render to screen
    # Maintain 60fps timing
    pass

uploader = MyOpenGLUploader()
processor = VideoFrameProcessor("video.mp4", gpu_uploader=uploader)

# Maintain real-time timing
fps = processor.decoder.fps
frame_duration = 1.0 / fps

for frame in processor.decoder:
    start = time.time()
    
    texture_id = uploader.upload_frame(frame)
    realtime_callback(frame, texture_id)
    uploader.release_texture(texture_id)
    
    # Sleep to maintain FPS
    elapsed = time.time() - start
    time.sleep(max(0, frame_duration - elapsed))
```

## Troubleshooting

### "No module named 'av'"
```bash
pip install av
```

### Slow decoding
- Check codec (H.265 is slower than H.264)
- Use hardware decoding (future feature)
- Reduce resolution

### Memory usage too high
- Reduce queue size
- Process frames immediately after decoding
- Don't accumulate frames in memory

### Timestamp issues
- VFR videos: Use PTS, not frame_index × (1/fps)
- CFR videos: Both PTS and index-based timing work

---

## Conclusion

This shared video core provides a clean, mode-agnostic foundation for video processing in KizunaSR. It handles the complex tasks of video decoding, timing, and frame management while remaining agnostic to downstream use (display vs encode).

The abstract GPU upload interface ensures compatibility with any graphics API (OpenGL, Vulkan, WebGPU) without coupling the core to specific implementations.

---
*Video Core Version: 1.0 MVP*  
*Status: Complete - Ready for Integration*

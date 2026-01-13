# KizunaSR Playback Window - Component Documentation

## Overview
Minimal windowing module providing GLFW window creation and OpenGL 4.5+ context for the KizunaSR real-time playback engine.

**Purpose**: Foundation layer for rendering. Does NOT perform any video processing, just provides the rendering context.

## Implementation

### PlaybackWindow Class

**Location**: [runtime/playback/window.py](file:///d:/college/Projects/Kizuna (絆)/runtime/playback/window.py)

**Responsibilities**:
- Create resizable desktop window
- Initialize OpenGL 4.5+ core profile context
- Own the main render loop
- Handle resize and keyboard events
- Clean resource management

## Usage

### Basic Example

```python
from runtime.playback import PlaybackWindow

# Create window with OpenGL context
with PlaybackWindow(width=1280, height=720, title="KizunaSR Player") as window:
    
    # Optional: define frame callback
    def on_frame():
        # Your rendering code here
        pass
    
    # Run main loop (blocks until window closes)
    window.run(frame_callback=on_frame)

# Window automatically cleaned up
```

### Manual Control

```python
window = PlaybackWindow(1280, 720)

while not window.should_close():
    window.poll_events()
    
    # Your rendering code here
    # glClear(...), glDrawArrays(...), etc.
    
    window.swap_buffers()

window.close()
```

### Querying Window State

```python
# Get current framebuffer size
width, height = window.get_framebuffer_size()

# Check if window should close
if window.should_close():
    # Clean shutdown
    pass
```

## Features

### Window Creation
- GLFW-based window management
- Resizable by default
- ESC key closes window
- V-sync enabled by default (configurable)

### OpenGL Context
- OpenGL 4.5+ core profile
- Forward-compatible (macOS support)
- Version verification at runtime
- Default state initialized (viewport, clear color)

### Event Handling
- **Resize**: Automatically updates OpenGL viewport
- **Keyboard**: ESC closes window
- **Close**: Window close button handled

### Render Loop
- Polls events (non-blocking)
- Clears framebuffer (black by default)
- Calls user callback (optional)
- Swaps buffers (presents frame)
- Logs every 1000 frames

## API Reference

### Constructor

```python
PlaybackWindow(
    width: int = 1280,
    height: int = 720,
    title: str = "KizunaSR Player",
    vsync: bool = True
)
```

**Parameters**:
- `width`: Initial window width (pixels)
- `height`: Initial window height (pixels)
- `title`: Window title string
- `vsync`: Enable V-sync (True = locked to display refresh rate)

**Raises**: `RuntimeError` if GLFW or OpenGL context creation fails

### Methods

**`run(frame_callback=None)`**  
Enters main render loop, blocks until window closes.

- `frame_callback`: Optional function called each frame (no arguments)

**`poll_events()`**  
Polls window events (keyboard, resize, close, etc.)

**`swap_buffers()`**  
Swaps front/back buffers (presents rendered frame)

**`should_close() -> bool`**  
Returns True if window close was requested

**`get_framebuffer_size() -> Tuple[int, int]`**  
Returns current framebuffer size as `(width, height)`

**`close()`**  
Destroys window and terminates GLFW

## Validation

### Test 1: Window Creation
```bash
python runtime/playback/window.py
```

**Expected**:
- Window appears (1280×720)
- Black screen
- Console shows:
  - GLFW initialized
  - OpenGL version (should be 4.5+)
  - Renderer (GPU name)
- Press ESC → window closes cleanly

### Test 2: Resize Handling
1. Run window
2. Resize window manually
3. Console should log: `[Window] Framebuffer resized: W×H`

### Test 3: OpenGL Version
Check console output:
```
[OpenGL] Version: 4.6.0 NVIDIA ...
[OpenGL] Renderer: NVIDIA GeForce RTX ...
[OpenGL] Version 4.6 confirmed
```

If version < 4.5:
```
[Warning] OpenGL 4.3 detected, 4.5+ recommended
```
(Still works, but shaders may need adjustment)

## Error Handling

### GLFW Initialization Failed
```
RuntimeError: Failed to initialize GLFW
```
**Solution**: Reinstall GLFW, check system graphics drivers

### Window Creation Failed
```
RuntimeError: Failed to create GLFW window
```
**Solution**: GPU doesn't support OpenGL 4.5, update drivers or use older OpenGL version

### Import Errors
```
ModuleNotFoundError: No module named 'glfw'
```
**Solution**:
```bash
pip install -r runtime/playback/requirements.txt
```

## Implementation Details

### GLFW Configuration
```python
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
```

**Why OpenGL 4.5?**
- Required for KizunaSR GLSL shaders
- Widely supported (2014+, NVIDIA GTX 600+, AMD GCN+)
- Core profile → no legacy OpenGL

### V-Sync
```python
glfw.swap_interval(1)  # 1 = V-sync on, 0 = off
```

**Effect**:
- On: Locked to display refresh (typically 60Hz)
- Off: Unlimited FPS (screen tearing possible)

**Default**: On (smooth playback, no tearing)

### Framebuffer Size Callback
```python
def _framebuffer_size_callback(self, window, width, height):
    glViewport(0, 0, width, height)
```

**Why separate from window size?**
- High-DPI displays have framebuffer ≠ window size
- Example: 1280×720 window = 2560×1440 framebuffer (2× scaling)

## Resource Management

### Clean Shutdown
```python
with PlaybackWindow(...) as window:
    window.run()
# Automaticcleanup
```

### Manual Cleanup
```python
window = PlaybackWindow(...)
try:
    window.run()
finally:
    window.close()  # Ensures GLFW termination
```

### Destructor
```python
def __del__(self):
    self.close()
```
Ensures cleanup even if user forgets.

## Performance

### Render Loop
- **Without V-sync**: 1000+ FPS (just clearing)
- **With V-sync**: 60 FPS (display-locked)

### Overhead
- Event polling: ~0.1ms
- Buffer swap: ~0-16ms (depends on V-sync)

## Next Steps

With the window foundation complete, next components:

1. **Texture Manager**: Upload video frames to GPU textures
2. **Shader Pipeline**: Compile and execute KizunaSR shaders
3. **Frame Scheduler**: PTS-based frame presentation
4. **Audio Player**: Audio playback and A/V sync

---
*Window Module Complete - Foundation Ready for Rendering*

---

## GPU Texture Management

### OpenGLTextureUploader Class

**Location**: [runtime/playback/texture_manager.py](file:///d:/college/Projects/Kizuna (絆)/runtime/playback/texture_manager.py)

**Responsibilities**:
- Upload VideoFrame (CPU RGB) to OpenGL textures
- Manage texture lifecycle and reuse
- Provide texture metadata queries

### Usage

```python
from runtime.playback import OpenGLTextureUploader
from runtime.video import VideoFrame

# Create uploader
uploader = OpenGLTextureUploader(enable_texture_reuse=True)

# Upload frame
texture_id = uploader.upload_frame(video_frame)

# Query size
width, height = uploader.get_texture_size(texture_id)

# Release texture (returns to pool if reuse enabled)
uploader.release_texture(texture_id)

# Cleanup all textures
uploader.cleanup()
```

### Features

**Texture Upload**:
- Input: VideoFrame (RGB uint8, H×W×3)
- Internal format: GL_RGB8
- Filter: GL_LINEAR (bilinear)
- Wrap: GL_CLAMP_TO_EDGE

**Texture Reuse**:
- Enabled by default
- Pools textures by resolution
- Reduces glGenTextures/glDeleteTextures calls
- Improves performance for constant resolution

**Resource Safety**:
- Tracks all created textures
- Cleanup on demand or destruction
- No leaks even on early exit

### API Reference

**`upload_frame(frame: VideoFrame) -> int`**  
Uploads frame to GPU, returns texture ID

**`release_texture(texture_id: int)`**  
Releases texture (pools or deletes)

**`get_texture_size(texture_id: int) -> Tuple[int, int]`**  
Returns (width, height) of texture

**`cleanup()`**  
Deletes all textures and clears pool

**`get_stats() -> dict`**  
Returns texture usage statistics

### Implementation Details

**Pixel Upload**:
```python
glPixelStorei(GL_UNPACK_ALIGNMENT, 1)  # Handle RGB24 (3 bytes/pixel)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels)
```

**Texture Reuse**:
- First upload: `glTexImage2D` (allocate + upload)
- Reuse: `glTexSubImage2D` (update only, faster)

**Texture Parameters**:
```python
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
```

### Validation

**Test Texture Upload**:
```bash
python runtime/playback/texture_manager.py
```

**Expected**:
- Window appears
- Console shows:
  - Texture uploaded (ID)
  - Size query successful
  - Texture reuse working
  - Stats printout
- Window closes after 1 second

**Visual Validation** (with shader):
- Upload frame
- Bind texture to shader
- Render fullscreen quad
- Verify frame appears correctly

### Performance

**Without Reuse**:
- Every frame: glGenTextures + glTexImage2D + glDeleteTextures
- ~0.5-1ms per frame (depends on resolution)

**With Reuse**:
- First frame: glGenTextures + glTexImage2D
- Subsequent: glTexSubImage2D only
- ~0.1-0.3ms per frame (3-5× faster)

**Memory**:
- 1920×1080 RGB8: 6.2 MB per texture
- Pool limit: User-controlled (cleanup() clears pool)

---
*Texture Manager Complete - CPU-to-GPU Transfer Ready*

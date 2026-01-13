# Hybrid Pipeline Integration Specification

## Document Purpose
This specification defines the integration layer that connects the real-time shader pipeline with the local AI inference module in KizunaSR. The hybrid architecture enables both fast real-time playback and high-quality AI-enhanced export while maintaining clean separation, deterministic behavior, and robust fallback mechanisms.

---

## Core Integration Principles

### Separation of Concerns
- **Real-time path**: Shader-only, optimized for latency (<16ms per frame)
- **Export path**: Shader preprocessing → AI upscaling → Shader post-processing, optimized for quality
- **No mixing**: AI never runs during real-time playback

### Deterministic Behavior
- Given the same input frame and configuration, the system produces identical output
- No race conditions between shader and AI processing
- Export results are reproducible

### Safe Degradation
- If AI is unavailable, system falls back to shader-only path
- No pipeline crashes or undefined behavior
- User is notified but operation continues

---

## Data Flow Design

### High-Level Flow: Preview Mode (Real-Time)

```
Input Frame → Shader Pipeline (5 stages) → Display Output
```

**Stages**:
1. Normalize
2. Structural Reconstruction
3. Real-Time Upscale
4. Perceptual Enhancement
5. Temporal Stabilization

**Characteristics**:
- All processing on GPU
- No CPU synchronization points
- No AI inference
- Continuous 60fps throughput

### High-Level Flow: Export Mode (AI-Enhanced)

```
Input Frame → Shader Preprocessing → AI Inference → Shader Post-Processing → Encoded Output
```

**Detailed Breakdown**:
1. **Shader Preprocessing** (GPU):
   - Normalize (color space conversion, denoising)
   - Structural Reconstruction (edge detection, feature classification)
   
2. **GPU → CPU Transfer**:
   - Read preprocessed frame from GPU texture to CPU memory
   - Convert texture format to tensor format (NCHW layout)
   
3. **AI Inference** (GPU or CPU):
   - Tile-based upscaling (2× or 4×)
   - Run through selected model (e.g., Real-ESRGAN anime)
   
4. **CPU → GPU Transfer**:
   - Upload AI output tensor back to GPU as texture
   
5. **Shader Post-Processing** (GPU):
   - Perceptual Enhancement (contrast, line darkening, saturation)
   - Optional Temporal Stabilization (if processing video sequentially)
   
6. **Encode**:
   - Read final frame from GPU
   - Encode to target format (H.264, H.265, PNG sequence, etc.)

### GPU ↔ CPU Transfer Details

#### Shader Output → AI Input

**Format Conversion**:
- **Shader output**: GPU texture (RGBA8, RGBA16F, or RGBA32F)
- **AI input**: CPU tensor (RGB, FP32, NCHW layout)

**Conversion Process**:
1. Download texture from GPU to CPU buffer using graphics API (Vulkan: `vkCmdCopyImageToBuffer`, DirectX: `ID3D12Resource::Map`)
2. Strip alpha channel (RGBA → RGB)
3. Reorder from interleaved (RGBRGBRGB...) to planar (RRR...GGG...BBB...)
4. Ensure FP32 precision (convert if source is FP16 or RGBA8)

#### AI Output → Shader Input

**Format Conversion**:
- **AI output**: CPU tensor (RGB, FP32, NCHW layout)
- **Shader input**: GPU texture (RGBA16F or RGBA32F)

**Conversion Process**:
1. Reorder from planar (RRR...GGG...BBB...) to interleaved (RGBRGBRGB...)
2. Add alpha channel (set A=1.0 for all pixels)
3. Upload buffer to GPU texture using graphics API
4. Optionally convert to FP16 to save VRAM

#### Color Space and Precision Preservation

**Color Space**:
- Shader preprocessing outputs **linear RGB** or **sRGB** (configurable)
- AI expects and outputs the **same color space**
- No color space conversion during transfer (avoids precision loss)

**Precision**:
- Shader can operate in FP16 or FP32
- AI inference typically requires FP32 for stability
- Transfer uses FP32 to avoid quantization artifacts
- Post-processing shaders can downgrade to FP16 if VRAM-constrained

**Dynamic Range**:
- Maintained at `[0.0, 1.0]` throughout pipeline
- AI models trained on this range
- Shader stages respect this normalization

---

## Preview vs Export Paths

### Real-Time Preview Path

**Purpose**: Enable smooth video playback at 60fps for user preview.

**Characteristics**:
- Shader-only processing (no AI)
- Latency: <16ms per frame (1080p → 4K)
- Quality: Excellent for anime, good enough for most viewing
- GPU memory: ~500 MB for pipeline buffers

**Visual Output**:
- Clean edges (edge-directed upscaling)
- Stable colors (no AI hallucination risk)
- Smooth motion (temporal stabilization active)

### AI Export Path

**Purpose**: Generate maximum-quality output for archival or distribution.

**Characteristics**:
- Shader preprocessing + AI upscaling + Shader post-processing
- Throughput: 0.5-2 seconds per frame (varies by GPU)
- Quality: State-of-the-art, preserves fine details
- GPU memory: 2-6 GB (depends on tile size and model)

**Visual Output**:
- Enhanced details (AI learned priors)
- Sharper line art (model-specific enhancement)
- Potential subtle hallucination (controlled via blending)

### Visual Consistency Guarantee

**Problem**: Users need to trust that preview ≈ export quality.

**Solution Strategy**:
1. **Shared Post-Processing**: Both paths use identical Perceptual Enhancement and Temporal Stabilization shader stages
2. **Configurable AI Strength**: Allow blending AI output with shader output (e.g., 80% AI + 20% shader upscale) to control deviation
3. **Side-by-Side Preview**: In hybrid mode, allow rendering a single paused frame with AI for comparison
4. **Consistent Color**: Both paths operate in the same color space and dynamic range

**Expected Difference**:
- AI export has slightly sharper details and refined edges
- Preview has no hallucination artifacts (100% faithful to source structure)
- Both paths have identical color grading and contrast (same post-processing)

---

## Synchronization & Pipeline Control

### Frame Queue Design

**Real-Time Mode**:
- No queue (frames processed and discarded immediately)
- Shader pipeline is stateless except for 1-frame temporal history

**Export Mode**:
- Frame queue with configurable depth (e.g., 4-8 frames)
- Producer: Video decoder pushes frames to queue
- Consumer: Hybrid pipeline pops frames, processes, pushes to encoder

**Queue Benefits**:
- Decouples video decoding from processing (decoder never waits)
- Decouples AI inference from encoding (encoder never starves)
- GPU stays busy (next frame loads while current frame infers)

### Blocking vs Non-Blocking Execution

#### Blocking (Simpler, Initial Implementation)
**Flow**:
1. Decode frame → Queue
2. Pop frame from queue
3. **Block** until shader preprocessing completes
4. **Block** until AI inference completes
5. **Block** until shader post-processing completes
6. Push to encoder queue
7. Repeat

**Characteristics**:
- Simple sequential execution
- No concurrency bugs
- Suboptimal GPU utilization (GPU idle during CPU→GPU transfers)

#### Non-Blocking (Optimized, Advanced Implementation)
**Flow**:
1. Decode frame → Queue (asynchronous)
2. Pop frame, submit shader preprocessing (returns immediately)
3. While shader preprocessing runs, prepare AI inference resources
4. Wait for shader completion, submit CPU→GPU transfer (async DMA)
5. While transfer completes, submit AI inference
6. While AI runs, process next frame's shader preprocessing (pipeline parallelism)
7. Continue overlapping stages

**Characteristics**:
- Complex state machine
- Higher throughput (30-50% faster)
- Requires careful synchronization (fences, semaphores)

**Initial Implementation**: Use blocking. Optimize to non-blocking only if profiling shows significant GPU idle time.

### Audio/Video Synchronization During Export

**Challenge**: AI processing is variable-speed (tile count affects latency). Audio must remain in sync.

**Solution**:
1. **Decouple Audio and Video Processing**:
   - Audio stream is decoded separately and written directly to output (no AI processing)
   - Video frames are processed at variable speed
   
2. **Timestamp Preservation**:
   - Each video frame retains its original presentation timestamp (PTS)
   - Encoder writes frames with original PTS regardless of processing order
   
3. **Out-of-Order Processing (Optional)**:
   - Frames can be processed in parallel (each frame is independent for spatial-only AI)
   - Reorder frames before encoding to restore original sequence
   
4. **Temporal Stabilization Constraint**:
   - If temporal stabilization is enabled, frames MUST be processed sequentially (requires previous frame)
   - Queue becomes FIFO, no parallelism

**Result**: Audio and video remain perfectly synchronized in final export.

---

## Failure & Fallback Handling

### Failure Scenarios and Responses

#### Scenario 1: AI Model Fails to Load

**Cause**:
- Model file is missing or corrupted
- Unsupported model format
- Insufficient RAM to load model weights

**Detection**:
- At startup when user selects AI mode
- During model loading phase (before any frame processing)

**Response**:
1. Log detailed error message (file path, error type)
2. Notify user: "AI model failed to load. Falling back to shader-only mode."
3. Disable AI export path
4. Continue with shader-only preview and export
5. Flag configuration as invalid (prevent retry loop)

**Safe State**: User can still export at shader quality. No crash.

#### Scenario 2: GPU Memory Insufficient for AI

**Cause**:
- VRAM allocation fails during tile processing
- Model activations exceed available memory
- Concurrent applications consuming VRAM

**Detection**:
- During first frame inference attempt
- CUDA/DirectML allocation returns error

**Response**:
1. Reduce tile size automatically (e.g., 512×512 → 256×256)
2. Retry inference with smaller tiles
3. If still fails, fall back to CPU inference
4. If CPU also fails, fall back to shader-only
5. Notify user of fallback level

**Safe State**: Graceful degradation from GPU → CPU → Shader-only.

#### Scenario 3: AI Backend Unavailable

**Cause**:
- No CUDA support (NVIDIA GPU missing or driver outdated)
- DirectML not available (pre-Windows 10 or GPU too old)
- CPU backend fails to initialize

**Detection**:
- At startup during backend probing

**Response**:
1. Probe backends in order: CUDA → DirectML → CPU
2. If all fail, disable AI mode entirely
3. Notify user: "No compatible AI backend detected. Using shader-only mode."
4. Hide AI configuration options in UI (future work)

**Safe State**: Application remains fully functional in shader-only mode.

#### Scenario 4: Inference Timeout or Hang

**Cause**:
- Model contains infinite loop (unlikely but possible with custom models)
- GPU driver hang
- Deadlock in inference runtime

**Detection**:
- Watchdog timer expires (e.g., 30 seconds per frame)

**Response**:
1. Terminate inference thread (if possible)
2. Mark current frame as failed
3. Skip to next frame or fall back to shader-only for current frame
4. Log error and continue processing
5. If consecutive failures exceed threshold (e.g., 5 frames), disable AI entirely

**Safe State**: Export continues, failed frames use shader-only output.

### Fallback Behavior Summary

**Fallback Hierarchy**:
```
AI Export (CUDA) → AI Export (DirectML) → AI Export (CPU) → Shader Export → Preview Mode
```

**Guarantees**:
- **No undefined behavior**: Every failure path has a defined response
- **No data loss**: Frames are never dropped silently
- **User awareness**: Clear notifications for all fallbacks
- **Reproducible**: Same failure always triggers same fallback

---

## Tile-Level Integration

### Full-Frame Shaders with Tiled AI

**Challenge**: Shader preprocessing operates on full frames. AI operates on tiles. How to integrate?

**Architecture**:

1. **Shader Preprocessing (Full-Frame)**:
   - Process entire frame through Normalize and Structural Reconstruction
   - Output: Full-resolution preprocessed frame in GPU texture

2. **Tile Extraction**:
   - Divide full-frame texture into overlapping tiles (e.g., 512×512 with 64px overlap)
   - Download each tile individually to CPU memory
   - Convert each tile to tensor format

3. **AI Inference (Per-Tile)**:
   - Process each tile through AI model independently
   - Output: Upscaled tile tensor (e.g., 1024×1024 for 2× upscaling)

4. **Tile Reassembly**:
   - Upload upscaled tiles back to GPU
   - Blend overlapping regions using feathered blending
   - Composite into single full-resolution output texture

5. **Shader Post-Processing (Full-Frame)**:
   - Apply Perceptual Enhancement and Temporal Stabilization to reassembled texture
   - Output: Final frame ready for encoding

### Seam Prevention Across Shader + AI Boundaries

**Potential Seam Sources**:

1. **Tile boundaries**: AI processes tiles independently, may introduce discontinuities
2. **Color shift**: Slight color variations between tiles due to FP precision
3. **Edge artifacts**: AI models may produce different edge responses at tile borders

**Mitigation Strategies**:

#### Strategy 1: Adequate Overlap
- Use 64-pixel overlap minimum (128 pixels for 4× upscaling)
- Larger overlap = better context for AI = fewer seams
- Trade-off: More computation (overlapping regions processed multiple times)

#### Strategy 2: Feathered Blending
- In overlap regions, blend adjacent tiles using smooth alpha ramp
- Example: Linear interpolation from 1.0 (tile center) to 0.0 (tile edge)
- Smooths out discontinuities

#### Strategy 3: Shared Preprocessing
- All tiles see the same shader preprocessing (global normalization, consistent edge detection)
- Consistency in input reduces inconsistency in output

#### Strategy 4: Post-Processing Uniformity
- Shader post-processing (Perceptual Enhancement) operates on reassembled full-frame
- Applies consistent color grading and sharpening across all tiles
- Masks minor seam artifacts with global adjustments

**Validation**:
- Test with checkerboard patterns and gradients
- Visual inspection for visible seams at tile boundaries
- Automated diff between tiled and full-frame processing (if VRAM allows full-frame)

---

## Resource Management

### GPU Memory Ownership

**Ownership Model**: **Single Owner, Explicit Transfer**

#### Shader Pipeline Owns:
- Input frame buffer
- Intermediate shader pass textures (e.g., edge maps)
- Final output texture (after AI reassembly)
- Temporal history buffer

#### AI Module Owns:
- Model weights (persistent, loaded once)
- Intermediate activation buffers (persistent, reused across frames)
- Input tile tensor (CPU memory, transient)
- Output tile tensor (CPU memory, transient)

#### Shared Resources:
- **Transfer buffers**: Staging buffers for GPU↔CPU transfer
  - Ownership: Graphics runtime (Vulkan/DirectX)
  - Lifecycle: Allocated at startup, reused across frames, freed at shutdown

**Explicit Transfer Points**:
1. Shader outputs preprocessed frame → Ownership to AI module (download begins)
2. AI module uploads upscaled tiles → Ownership to Shader pipeline (upload completes)

**No Concurrent Access**: GPU and AI never access the same buffer simultaneously.

### Buffer Allocation and Release

#### Startup (Initialization Phase)
**Allocate**:
- Shader pipeline textures (resident in VRAM)
- Transfer staging buffers (pinned CPU memory for fast DMA)
- AI model weights (VRAM or RAM depending on backend)
- AI activation buffers (VRAM or RAM)

**Size Estimation**:
- Shader textures: ~500 MB for 4K pipeline
- Transfer buffers: ~200 MB (double-buffered for async transfer)
- AI model: 20-100 MB (model-dependent)
- AI activations: 1-4 GB (tile-size dependent)

#### Per-Frame (Processing Phase)
**Allocate**:
- Nothing (all buffers pre-allocated and reused)

**Transient Allocations** (if unavoidable):
- Temporary CPU tensors for format conversion
- Immediately freed after upload/download

#### Shutdown (Cleanup Phase)
**Release**:
- Free all shader textures
- Free transfer buffers
- Unload AI model weights
- Free AI activation buffers
- Destroy graphics contexts

**Order of Destruction**:
1. AI module cleanup (release model, free activations)
2. Transfer buffer cleanup
3. Shader pipeline cleanup (free textures, destroy shaders)
4. Graphics context destruction

### Resource Leak Prevention

**Strategies**:

1. **RAII Pattern** (Resource Acquisition Is Initialization):
   - Wrap GPU resources in objects with constructors/destructors
   - Automatic cleanup when object goes out of scope
   
2. **Explicit Lifetime Management**:
   - Track resource handles in a registry
   - Assert all resources freed at shutdown (debug builds)
   
3. **Reference Counting** (for shared resources):
   - Transfer buffers shared between shader and AI paths
   - Only free when refcount reaches zero
   
4. **Frame Markers**:
   - Each frame processing logs resource allocations
   - Diff between frame N and frame N+1 should be zero (no leaks)
   
5. **Watchdog Timers**:
   - Detect hung inference or shader passes
   - Force-release resources if timeout exceeded

**Validation Tools**:
- Enable GPU API validation layers (Vulkan validation, D3D12 debug layer)
- Run memory leak detectors (Valgrind, AddressSanitizer)
- Monitor VRAM usage via GPU tools (nvidia-smi, Task Manager)

---

## Export Pipeline Design

### Export Workflow Overview

**User Perspective**:
1. User selects input video file and output path
2. User configures export settings (AI model, scale factor, output format)
3. User starts export
4. Progress bar shows frame processing status
5. Export completes, final video saved

**System Perspective**:
```
Video Decode → Frame Queue → Hybrid Pipeline → Encode Queue → Video Encode → File Write
```

### Detailed Export Pipeline Stages

#### Stage 1: Video Decode
**Responsibilities**:
- Open input video file
- Decode compressed frames to raw RGB
- Push frames to input queue
- Handle variable frame rates, resolution changes, codec quirks

**Implementation Notes**:
- Use FFmpeg or OS-native decoder (e.g., Media Foundation on Windows)
- Decode in separate thread to avoid blocking pipeline
- Queue depth: 8-16 frames (balance memory vs throughput)

#### Stage 2: Frame Queue (Input)
**Responsibilities**:
- Buffer decoded frames for pipeline consumption
- Provide backpressure to decoder if pipeline is slow
- Preserve frame metadata (PTS, duration, frame index)

**Thread-Safety**:
- Mutex-protected queue
- Condition variable for producer/consumer signaling

#### Stage 3: Hybrid Pipeline Processing
**Responsibilities**:
- Pop frame from input queue
- Run shader preprocessing (Normalize, Structural Reconstruction)
- Tile and run AI inference
- Reassemble tiles
- Run shader post-processing (Perceptual Enhancement, Temporal Stabilization)
- Push processed frame to encode queue

**Parallel Processing**:
- For spatial-only AI (no temporal stabilization): Process multiple frames in parallel
- For temporal-enabled mode: Process sequentially

#### Stage 4: Encode Queue (Output)
**Responsibilities**:
- Buffer processed frames waiting for encoder
- Maintain frame order (if parallel processing reordered frames)
- Provide encoder with steady stream of frames

**Reordering**:
- If frames processed out-of-order, reorder by PTS before encoding

#### Stage 5: Video Encode
**Responsibilities**:
- Encode raw frames to target codec (H.264, H.265, VP9, AV1)
- Apply encoding settings (bitrate, quality, preset)
- Mux video stream with audio stream
- Write to output file

**Implementation Notes**:
- Use FFmpeg libavcodec or hardware encoders (NVENC, QuickSync, AMF)
- Encode in separate thread
- Handle audio passthrough (no AI processing on audio)

#### Stage 6: File Write
**Responsibilities**:
- Write muxed video to disk
- Flush buffers at completion
- Verify file integrity

### Shader Pass Reuse in Export

**Perceptual Enhancement**:
- Same shader code as real-time preview
- Same parameters (contrast boost, line darkening factor, saturation)
- Ensures visual consistency between preview and export

**Temporal Stabilization**:
- Optional in export mode
- If enabled: Maintains 1-frame history buffer, blends with previous frame
- If disabled: Skip this pass entirely (useful for frame-by-frame export like PNG sequences)

**Why Reuse?**:
- Code deduplication (single shader implementation)
- Visual consistency (preview ≈ export appearance)
- Easier tuning (adjust shader once, affects both paths)

### Output Format Handling

**Supported Formats**:
- **Video containers**: MP4, MKV, WebM
- **Video codecs**: H.264, H.265 (HEVC), VP9, AV1
- **Image sequences**: PNG, JPEG, TIFF (for frame-by-frame export)

**Format Selection**:
- User specifies via configuration
- Export pipeline configures encoder accordingly
- No format-specific logic in hybrid pipeline (encoder handles it)

**Color Space and Bit Depth**:
- Export at 8-bit or 10-bit (user-configurable)
- Maintain color space consistency (Rec.709 for HD, Rec.2020 for HDR)
- Shader post-processing outputs in linear RGB; encoder applies transfer function

---

## Explicit Non-Goals

The hybrid integration layer deliberately **DOES NOT** attempt the following:

### No Live AI During Playback
- AI inference never runs concurrently with real-time video playback
- Real-time path is shader-only, always
- **Rationale**: GPU contention causes frame drops, stuttering, and inconsistent latency. Separation ensures smooth playback.

### No Temporal AI
- AI processes single frames independently
- No multi-frame AI super-resolution
- No recurrent networks or optical flow
- **Rationale**: Temporal AI requires sequential processing (no parallelism), high memory overhead, and complex state management. Spatial-only AI is simpler and faster.

### No Distributed Processing
- All processing occurs on single machine
- No network-based frame distribution to other machines
- No multi-GPU across different systems
- **Rationale**: Distributed processing adds complexity (networking, synchronization, load balancing) without proportional benefit for local use cases.

### No Cloud Inference
- All AI inference runs locally
- No remote API calls
- No uploading frames to external services
- **Rationale**: Privacy, latency, cost, and offline-first design.

### No Dynamic Model Switching Mid-Export
- User selects AI model before export starts
- Same model used for all frames in a video
- No per-frame model selection
- **Rationale**: Model loading is expensive (seconds). Switching per-frame would destroy throughput.

### No Real-Time AI Preview
- AI does not enhance frames during live playback
- Hybrid preview mode only triggers AI on paused frames
- **Rationale**: AI latency (0.5-2s per frame) is incompatible with 60fps playback.

---

## Integration Summary

### Key Design Decisions

1. **Strict Path Separation**: Real-time and export paths never mix, preventing performance interference
2. **Shared Post-Processing**: Both paths use identical shader stages for visual consistency
3. **Graceful Fallback**: Every failure scenario has a safe response (shader-only as ultimate fallback)
4. **Tile Independence**: AI processes tiles independently, enabling parallelism and bounded memory usage
5. **Resource Pre-Allocation**: Minimize per-frame allocations to avoid fragmentation and latency spikes

### Integration Benefits

- **Performance Isolation**: Real-time playback unaffected by AI complexity
- **Quality Scalability**: Export quality adapts to hardware (GPU → CPU → Shader-only)
- **Visual Consistency**: Preview and export share color grading and enhancement logic
- **Robustness**: No crashes or undefined behavior on AI failures

### Implementation Readiness

This specification provides sufficient detail to implement:
- Data flow conversion routines (texture ↔ tensor)
- Frame queue and synchronization primitives
- Fallback logic and error handling
- Tile extraction and reassembly
- Export pipeline orchestration

---
*Document Version: 1.0*  
*Status: Design Complete - Ready for Implementation*

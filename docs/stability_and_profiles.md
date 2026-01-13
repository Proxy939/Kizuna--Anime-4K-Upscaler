# Stability and Profiles Specification

## Document Purpose
This specification defines the stability, safety, and profile management layer of KizunaSR. It ensures reliable behavior across diverse hardware configurations, content types, and execution modes through automatic capability detection, safe fallback mechanisms, and deterministic execution.

---

## Core Stability Principles

### Graceful Degradation
The system always provides a working output path. If AI fails, fall back to shader-only. If GPU fails, fall back to CPU. No undefined behavior or crashes.

### Deterministic Execution
Given the same input content, hardware configuration, and profile selection, the system produces identical output every time. No randomness, no race conditions.

### Fail-Safe by Default
All operations have defined failure paths. Missing models, insufficient VRAM, or unsupported hardware trigger safe fallbacks with user notification, not silent failures or crashes.

### Resource Discipline
Strict buffer management prevents memory leaks, VRAM exhaustion, and runaway allocations. All resources have explicit lifetimes.

---

## Content Analysis & Auto-Profiling

### Input Content Analysis

**Purpose**: Understand the characteristics of input video/images to select appropriate processing profiles automatically.

**Analysis Dimensions**:

#### 1. Resolution
- **Detection**: Read video metadata (width × height)
- **Categories**:
  - SD (≤720p): Lightweight processing, smaller tiles
  - HD (1080p): Standard processing
  - UHD (4K+): Large tiles, careful VRAM management
  
- **Impact on Profile**:
  - SD content: Can afford larger upscale factors (4×) without excessive memory
  - UHD content: May limit upscale to 2× or force CPU fallback

#### 2. Bitrate and Compression Level
- **Detection**: Analyze bitrate metadata, sample frames for compression artifacts
- **Categories**:
  - High bitrate (>8 Mbps for 1080p): Clean source, minimal preprocessing needed
  - Medium bitrate (2-8 Mbps): Standard compression, moderate denoising
  - Low bitrate (<2 Mbps): Heavy artifacts, aggressive preprocessing required
  
- **Impact on Profile**:
  - Low bitrate: Enable stronger denoising in Normalize stage
  - High bitrate: Skip denoising to preserve detail

#### 3. Frame Rate
- **Detection**: Read FPS metadata (23.976, 24, 30, 60, etc.)
- **Categories**:
  - Film rate (23.976-24): Typical for anime films
  - Standard (30): Common for TV anime
  - High framerate (60+): Modern anime, action sequences
  
- **Impact on Profile**:
  - High framerate: Temporal stabilization more critical (more frames to stabilize)
  - Film rate: Can afford heavier per-frame processing (fewer fps)

#### 4. Anime-Specific Trait Detection
- **Detection**: Sample frames and analyze:
  - Edge density (line art prevalence)
  - Color histogram (flat vs gradated regions)
  - Texture variance (cel-shaded vs detailed backgrounds)
  
- **Categories**:
  - Pure cel-shaded anime: High edge density, low texture variance
  - Hybrid style: Mix of cel and detailed backgrounds
  - Non-anime: Low edge density, high texture variance
  
- **Impact on Profile**:
  - Pure anime: Aggressive edge-directed upscaling, minimal texture processing
  - Non-anime: Fall back to more generic upscaling (or warn user of suboptimal results)

### Automatic Profile Selection

**Selection Algorithm** (Conceptual):
1. Analyze first 10 frames of input content
2. Classify resolution, bitrate, framerate, style
3. Detect available hardware capabilities (GPU, VRAM, CPU)
4. Match content requirements to hardware capabilities
5. Select highest-quality profile that hardware can sustain
6. If no safe profile available, fall back to shader-only + warn user

**Example Decision Tree**:
- **If** UHD + Low VRAM (<6 GB) → **Profile**: Shader-only or 2× AI with small tiles
- **If** HD + NVIDIA GPU + High bitrate → **Profile**: AI Export 4× (CUDA)
- **If** SD + CPU-only → **Profile**: Shader-only (CPU cannot run AI at acceptable speed)

**User Override**: Allow manual profile selection to override automatic choice (advanced users may accept long processing times).

---

## Processing Profiles

### Profile 1: Real-Time Playback (Shader-Only)

**Purpose**: Smooth 60fps video playback for live viewing.

**Enabled Components**:
- Shader pipeline: All 5 stages (Normalize → Structural Reconstruction → Real-Time Upscale → Perceptual Enhancement → Temporal Stabilization)
- Temporal stabilization: Active (1-frame history)

**Disabled Components**:
- AI inference: Completely disabled
- Tile processing: Not applicable
- Export encoding: Not applicable

**Performance Guarantees**:
- Latency: <16ms per frame (1080p → 4K)
- Throughput: 60fps sustained
- VRAM usage: <1 GB

**Use Case**: User watches anime in media player with KizunaSR real-time upscaling filter.

**Safety Constraints**:
- No blocking operations (all GPU-resident)
- No CPU synchronization points
- Bounded memory allocation (fixed buffer sizes)

---

### Profile 2: AI Export (Quality Mode)

**Purpose**: Maximum quality offline export for archival or distribution.

**Enabled Components**:
- Shader preprocessing: Normalize + Structural Reconstruction
- AI inference: Tile-based, CUDA/DirectML/CPU (auto-selected)
- Tile processing: Overlap + feathered blending
- Shader post-processing: Perceptual Enhancement + Temporal Stabilization (optional)
- Export encoding: H.264/H.265/VP9/AV1

**Disabled Components**:
- Real-time temporal constraints (no 16ms deadline)

**Performance Characteristics**:
- Throughput: 0.5-2 seconds per frame (GPU-dependent)
- VRAM usage: 2-6 GB (tile size adaptive)
- Export speed: 30-60 frames per minute on RTX 3060

**Use Case**: User exports anime episode at 4K for Blu-ray or streaming upload.

**Safety Constraints**:
- Adaptive tile sizing (reduce if VRAM insufficient)
- Fallback to CPU if GPU fails
- Watchdog timer (30s per frame timeout)

---

### Profile 3: Batch Processing (Optimized Export)

**Purpose**: Process multiple files or entire series efficiently.

**Enabled Components**:
- Same as AI Export profile
- Additional optimizations:
  - Parallel frame processing (if temporal disabled)
  - Multi-file queue
  - Persistent model loading (load once, process many files)

**Performance Characteristics**:
- Throughput: 10-20% faster than AI Export (due to batching)
- VRAM usage: Same as AI Export
- Overhead reduction: No model reloading between files

**Use Case**: User exports entire anime season (12-24 episodes) overnight.

**Safety Constraints**:
- Per-file error isolation (one file failure doesn't abort batch)
- Memory cleanup between files (prevent leak accumulation)
- Progress checkpointing (resume from last successful file after crash)

---

### Profile 4: Hybrid Preview (Paused Frame AI)

**Purpose**: Real-time playback with on-demand AI enhancement of paused frames.

**Enabled Components**:
- During playback: Shader-only (same as Profile 1)
- When paused: AI inference on current frame only
- Display: Show AI-enhanced frame while paused, revert to shader on resume

**Performance Characteristics**:
- Playback: 60fps (shader-only)
- Pause-to-AI latency: 0.5-2 seconds (GPU-dependent)

**Use Case**: User previews AI quality during playback without full export.

**Safety Constraints**:
- AI inference never blocks playback thread
- Async inference (user can unpause before AI completes, result discarded)
- No VRAM contention (shader and AI share GPU but not memory regions)

---

### Profile Selection Safety Rules

**Rule 1: Hardware Compatibility**
- If no CUDA/DirectML/CPU backend available → Force Profile 1 (Shader-only)

**Rule 2: VRAM Headroom**
- If VRAM < 4 GB → Disable AI profiles or force small tiles
- If VRAM < 2 GB → Force Profile 1 (Shader-only)

**Rule 3: Content Resolution**
- If input is 8K → Force 2× upscale (not 4×) or Profile 1

**Rule 4: User Constraints**
- If user requests real-time → Force Profile 1 regardless of hardware

**Rule 5: Model Availability**
- If selected AI model missing/invalid → Fall back to Profile 1

---

## Hardware Capability Detection

### GPU Detection

**Detection Process**:
1. Enumerate available GPUs via graphics API (Vulkan: `vkEnumeratePhysicalDevices`, DirectX: `IDXGIFactory::EnumAdapters`)
2. Query device properties:
   - Vendor (NVIDIA, AMD, Intel)
   - Model (e.g., RTX 3060, RX 6600)
   - VRAM capacity
   - Compute capability (CUDA compute level, DirectX feature level)
   - Precision support (FP16, FP32, INT8)

**Capability Classification**:
- **High-end** (RTX 4060+, RX 7600+): 8+ GB VRAM, full FP16/FP32, run largest tiles (768×768)
- **Mid-range** (RTX 3050-3060, RX 6600): 4-8 GB VRAM, FP16/FP32, standard tiles (512×512)
- **Low-end** (GTX 1650, integrated GPUs): 2-4 GB VRAM, FP32 only, small tiles (256×256)
- **Unsupported**: <2 GB VRAM or no compute support → CPU fallback or shader-only

### CUDA/DirectML Backend Probing

**CUDA Availability**:
- Check for NVIDIA GPU
- Query CUDA runtime version (`cudaGetDeviceCount()`)
- Verify compute capability ≥ 6.0 (Pascal or newer for FP16)
- If all pass → CUDA backend available

**DirectML Availability**:
- Check for DirectX 12-capable GPU
- Query DirectML library presence
- Verify feature level ≥ 1.0
- If all pass → DirectML backend available

**CPU Fallback**:
- Always available (ONNX Runtime CPU or OpenVINO)
- No special detection needed

**Backend Priority**: CUDA > DirectML > CPU (for NVIDIA), DirectML > CPU (for AMD/Intel)

### VRAM Estimation

**Static Estimation**:
- Query total VRAM: `vkGetPhysicalDeviceMemoryProperties()` or `GetVideoMemoryInfo()`
- Reserve 1 GB for OS/other applications
- Available VRAM = Total - Reserved

**Dynamic Monitoring**:
- Before allocating large buffers (AI model, tiles), check current usage
- If allocation would exceed 90% of total VRAM → Reduce tile size or fall back
- Poll VRAM usage every 10 frames during processing (detect external pressure)

**Allocation Strategy**:
- Shader buffers: 500 MB (fixed)
- AI model weights: 20-100 MB (model-dependent)
- AI activations: 512 MB to 4 GB (tile size adaptive)
- Transfer buffers: 200 MB (fixed)
- **Total estimate**: 1-5 GB depending on profile and tile size

### Precision Support Detection

**FP16 (Half Precision)**:
- Query GPU shader capabilities (Vulkan: `VkPhysicalDeviceShaderFloat16Int8Features`)
- If supported → Use FP16 for shader pipeline (2× bandwidth reduction)
- If not → Use FP32 (no functional difference, just slower)

**Impact on AI**:
- AI models typically require FP32 for stability
- FP16 AI inference possible but requires model quantization (advanced feature)

### Safe Fallback on Capability Mismatch

**Scenario 1: GPU Too Weak**
- If VRAM < minimum for selected profile → Auto-downgrade to lower profile

**Scenario 2: No GPU Compute Support**
- If GPU exists but lacks compute (old integrated GPUs) → CPU fallback

**Scenario 3: Driver Issues**
- If API initialization fails (outdated drivers) → Notify user, suggest driver update, fall back to CPU or shader-only

---

## Error Handling & Fail-Safe Design

### Error Categories and Responses

#### Category 1: Initialization Errors

**Error**: AI model file not found or corrupted
- **Detection**: At startup during model loading
- **Response**:
  1. Log error with file path and expected location
  2. Notify user: "AI model not found. Using shader-only mode."
  3. Disable AI profiles
  4. Continue with shader-only
- **Safe State**: Shader-only functional

**Error**: GPU driver initialization fails
- **Detection**: At startup during graphics context creation
- **Response**:
  1. Log error with driver version and error code
  2. Notify user: "GPU initialization failed. Check drivers."
  3. Fall back to CPU-only processing (very slow)
- **Safe State**: CPU-based shader rendering + CPU AI (if enabled)

**Error**: Insufficient VRAM for minimum profile
- **Detection**: At startup after VRAM query
- **Response**:
  1. Log available vs required VRAM
  2. Notify user: "Insufficient VRAM. Using reduced quality."
  3. Force smallest tile size or shader-only
- **Safe State**: Degraded but functional

#### Category 2: Runtime Errors

**Error**: AI inference fails mid-frame
- **Detection**: CUDA/DirectML runtime error during tile processing
- **Response**:
  1. Log error and frame index
  2. Fall back to shader upscale for current frame only
  3. Retry AI on next frame (transient error tolerance)
  4. If 5 consecutive failures → Disable AI for remainder
- **Safe State**: Mixed output (some AI frames, some shader frames)

**Error**: GPU memory allocation fails during processing
- **Detection**: VRAM allocation returns null/error
- **Response**:
  1. Log allocation size and available VRAM
  2. Reduce tile size by 50% (512×512 → 256×256)
  3. Retry allocation
  4. If still fails → Fall back to CPU or shader-only
- **Safe State**: Reduced quality or CPU processing

**Error**: Shader compilation fails
- **Detection**: At runtime during pipeline initialization
- **Response**:
  1. Log shader source and compile error
  2. This is critical (no shader = no output)
  3. Abort processing with clear error message
  4. Suggest GPU driver update
- **Safe State**: Cannot recover (shader is mandatory)

#### Category 3: Content Errors

**Error**: Input video codec unsupported
- **Detection**: During video decode initialization
- **Response**:
  1. Log codec type and error
  2. Notify user: "Unsupported codec. Try converting to H.264."
  3. Abort processing
- **Safe State**: No processing (input invalid)

**Error**: Input resolution exceeds maximum (e.g., 16K)
- **Detection**: During content analysis
- **Response**:
  1. Log input resolution
  2. Notify user: "Resolution too large. Maximum 8K supported."
  3. Offer to downsample or abort
- **Safe State**: User decision required

### Fail-Safe Design Patterns

#### Pattern 1: Defensive Validation
- **Principle**: Validate all inputs before processing
- **Application**: Check model file checksum, verify VRAM before allocation, validate frame dimensions

#### Pattern 2: Stateless Fallback
- **Principle**: Shader-only path has no dependencies
- **Application**: If AI or hybrid fails, shader-only always works (no external state)

#### Pattern 3: Error Isolation
- **Principle**: Failures in one component don't propagate
- **Application**: AI failure doesn't crash shader pipeline; single frame failure doesn't abort export

#### Pattern 4: Progressive Degradation
- **Principle**: System tries multiple fallback levels before failing
- **Application**: GPU AI → CPU AI → Shader-only (3 tiers)

---

## Resource & Memory Safety

### GPU Resource Management

#### Buffer Lifetime Model

**Persistent Buffers** (Allocated once, reused across frames):
- Shader pipeline textures (input, intermediate, output)
- Transfer staging buffers
- AI model weights
- AI activation buffers

**Transient Buffers** (Allocated per-operation, freed immediately):
- Temporary CPU tensors for format conversion
- Per-tile upload/download buffers (if not using staging pool)

**Lifetime Tracking**:
- Each buffer has an explicit owner (shader pipeline, AI module, transfer manager)
- Buffers are reference-counted if shared
- Destruction follows strict order (AI → Transfer → Shader → Context)

#### Allocation Strategy

**Pre-Allocation** (Preferred):
- Allocate all persistent buffers at startup based on selected profile
- Reuse buffers across frames (zero per-frame allocation)
- Benefits: No fragmentation, predictable memory usage, no allocation latency

**Dynamic Allocation** (Fallback):
- If content resolution changes mid-processing → Reallocate buffers
- If tile size adapts due to VRAM pressure → Reallocate activation buffers
- Always free old buffers before allocating new ones

#### VRAM Exhaustion Prevention

**Monitoring**:
- Poll available VRAM every 10 frames
- If free VRAM < 1 GB → Trigger warning
- If free VRAM < 500 MB → Force tile size reduction

**Adaptive Tile Sizing**:
- Start with largest tile size hardware can support
- If allocation fails → Reduce tile size by 25%
- Repeat until allocation succeeds or reach minimum (256×256)
- If minimum fails → Fall back to CPU

**External Pressure Handling**:
- If other applications consume VRAM during processing → Detect via VRAM polling
- Pause processing, free non-essential buffers (e.g., temporal history), retry
- If still fails → Notify user of VRAM contention, suggest closing other applications

### CPU Memory Management

**Tensor Buffers**:
- Allocated for AI input/output tiles
- Size: ~50 MB per tile (512×512 RGB FP32)
- Freed immediately after GPU upload/download (not persistent)

**Video Decode Buffers**:
- Decoder maintains internal frame buffer pool
- KizunaSR does not control decoder memory directly
- Limit decode queue depth to prevent RAM exhaustion (max 16 frames)

**Leak Prevention**:
- Use RAII wrappers for all allocations (smart pointers in C++, context managers in Python)
- Track allocations in debug builds (leak detector)
- Periodic memory audits (compare frame N vs frame N+1000, should be identical)

---

## Logging & Debugging Strategy

### Log Levels

#### INFO
**Purpose**: General operational information
**Examples**:
- "Loaded AI model: realESRGAN_anime_2x.onnx"
- "Selected profile: AI Export (CUDA)"
- "Processing frame 100/1000"

**Frequency**: Sparse (startup, profile selection, major milestones)

#### WARNING
**Purpose**: Non-fatal issues or degraded operation
**Examples**:
- "VRAM below 1 GB, reducing tile size"
- "AI inference slow (2.5s/frame), consider using smaller model"
- "Falling back to CPU backend"

**Frequency**: Occasional (when fallback triggered or performance degrades)

#### ERROR
**Purpose**: Failures requiring fallback or user action
**Examples**:
- "AI model failed to load: file not found"
- "GPU memory allocation failed"
- "Frame decode error at timestamp 00:05:23"

**Frequency**: Rare (only on actual failures)

#### DEBUG
**Purpose**: Detailed diagnostics for developers
**Examples**:
- "Tile 5/16: inference time 120ms"
- "Shader pass timings: Normalize 2ms, Upscale 8ms"
- "VRAM usage: 3.2 GB / 8 GB"

**Frequency**: Verbose (per-frame or per-operation), disabled by default

### Log Output Destinations

**Console** (stdout/stderr):
- INFO, WARNING, ERROR levels
- Real-time feedback during processing
- Suitable for users monitoring progress

**File** (kizuna_sr.log):
- All levels including DEBUG
- Rolling log file (max 10 MB, keep last 3 files)
- Structured format (timestamp, level, component, message)

**Example Log Entry**:
```
[2026-01-13 18:40:12] [INFO] [AIModule] Loaded model: realESRGAN_anime_2x.onnx (52 MB)
[2026-01-13 18:40:15] [WARN] [ResourceManager] VRAM usage high: 7.2/8 GB, reducing tile size
[2026-01-13 18:40:20] [ERROR] [ShaderPipeline] Frame 42 processing failed: CUDA error 2
```

### Debugging Support

**Frame Dump Mode**:
- When enabled, save intermediate outputs to disk:
  - Preprocessed frame (after shader stages 1-2)
  - Per-tile AI input tensors
  - Per-tile AI output tensors
  - Final composited output
- Useful for diagnosing seam artifacts or color shifts

**Performance Profiling**:
- Per-stage timing logs (DEBUG level)
- GPU profiler integration (NVIDIA Nsight, AMD Radeon Profiler)
- Allows identifying bottlenecks without code instrumentation

**Reproducibility Metadata**:
- Log full configuration at start of processing:
  - KizunaSR version
  - Selected profile
  - Hardware details (GPU model, VRAM, CPU)
  - AI model name and checksum
  - Input file checksum
- Enables exact reproduction of results for research or bug reports

---

## Determinism & Reproducibility

### Sources of Non-Determinism (and Mitigation)

#### GPU Floating-Point Precision
**Issue**: GPU may use different precision or rounding modes across runs
**Mitigation**:
- Use FP32 for AI inference (more stable than FP16)
- Avoid `--fast-math` compiler flags (non-deterministic)
- Prefer deterministic GPU operations where available

#### Tile Processing Order
**Issue**: Tiles processed in parallel may complete in different orders
**Mitigation**:
- Reassemble tiles in deterministic order (sorted by tile index)
- Sequential blending (process overlap regions left-to-right, top-to-bottom)

#### Random Number Generation
**Issue**: Some AI models use dropout or stochastic layers
**Mitigation**:
- **Enforce inference mode**: All models run in eval mode (no dropout)
- **No generative models**: Exclude diffusion or GAN-based SR (non-deterministic)
- **Fixed seed**: If any randomness required (e.g., dithering), use fixed seed

#### Memory Allocation Addresses
**Issue**: Pointer addresses vary across runs (affects hash-based caching)
**Mitigation**:
- Don't rely on pointer values for cache keys
- Use content-addressable caching (hash frame data, not pointers)

### Configuration Tracking

**Config File Snapshot**:
- At start of export, save complete configuration to JSON:
  - Profile name
  - AI model path and checksum
  - Tile size, overlap amount
  - Shader parameters (contrast, saturation, sharpening)
  - Hardware used (GPU model, backend)
  
**Embedding in Output**:
- Store config JSON as metadata in exported video (if format supports)
- Allows future verification: "Was this video processed with the same settings?"

### Reproducibility Validation

**Self-Test Mode**:
- Process same input frame twice
- Compare outputs pixel-by-pixel (or with PSNR/SSIM)
- If difference detected → Log warning and investigation needed

**Regression Testing**:
- Maintain reference output frames for test videos
- After code changes, reprocess test videos
- Compare new outputs to references (should be identical for deterministic pipeline)

**Research Use Case**:
- Researchers can publish KizunaSR config files alongside papers
- Others can reproduce exact upscaling results for comparisons

---

## Explicit Non-Goals

The stability layer deliberately **DOES NOT** attempt the following:

### No Performance Benchmarking
- System does not auto-benchmark hardware
- No FPS counters or performance metrics in production builds
- **Rationale**: Benchmarking adds overhead and complexity. Profiling tools (external) serve this purpose better.

### No UI-Level Error Handling
- No graphical error dialogs or progress bars
- Errors logged to console/file only
- **Rationale**: KizunaSR is a library/backend. UI integration is a separate layer.

### No Cloud Monitoring or Telemetry
- No automatic crash reports
- No anonymous usage statistics
- No phone-home functionality
- **Rationale**: Privacy-first design. Users control all data.

### No Self-Healing AI Logic
- System does not auto-download missing models
- No automatic model retraining or fine-tuning
- No adaptive algorithm selection based on output quality
- **Rationale**: Automation introduces complexity and unpredictability. Explicit configuration preferred.

### No Dynamic Code Loading
- No runtime plugin system
- No JIT compilation of shaders based on content
- **Rationale**: Determinism and security. Static configuration only.

### No Multi-Tenancy
- System designed for single user, single task at a time
- No concurrent processing of multiple videos by different users
- **Rationale**: Resource contention and complexity. Batching handles multiple files sequentially.

---

## Stability Summary

### Key Design Decisions

1. **Auto-Profiling**: Intelligent content analysis selects appropriate processing profile automatically
2. **Graceful Fallback**: Four-tier degradation (CUDA → DirectML → CPU → Shader-only)
3. **Resource Pre-Allocation**: Minimize runtime allocations to prevent leaks and fragmentation
4. **Deterministic Execution**: Reproducible results for research and validation
5. **Comprehensive Logging**: DEBUG/INFO/WARN/ERROR levels for diagnostics without performance impact

### Stability Guarantees

- **No Undefined Behavior**: Every error path has explicit handling
- **No Silent Failures**: All failures logged and user-notified
- **No Crashes**: Fallback mechanisms ensure operation continues
- **No Memory Leaks**: RAII and lifecycle tracking prevent resource leaks
- **Reproducible Results**: Same input + config → same output

### Integration with Other Specifications

- **Shader Pipeline**: Stability layer enforces safe shader parameter ranges
- **AI Module**: Stability layer handles model loading failures and backend selection
- **Hybrid Integration**: Stability layer manages resource allocation for shader + AI

---
*Document Version: 1.0*  
*Status: Design Complete - Ready for Implementation*

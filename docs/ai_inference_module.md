# AI Inference Module Specification

## Document Purpose
This specification defines the architecture and design of the local AI inference subsystem for KizunaSR. The AI module provides optional, offline super-resolution capabilities that complement the real-time shader pipeline without interfering with playback performance.

---

## Core Principles

### Local-First Philosophy
The AI inference module operates **entirely on the user's local hardware**. No data is transmitted to external servers, no cloud APIs are called, and no internet connection is required for inference.

### Offline Execution Model
AI inference is designed for:
- **Export mode**: Processing video files frame-by-frame for maximum quality output
- **Batch processing**: Upscaling multiple frames or entire videos
- **Hybrid mode**: Optionally enhancing paused frames during playback

AI inference **does not run** during real-time video playback. The shader pipeline handles real-time scenarios exclusively.

---

## AI Execution Model

### Hardware Utilization

#### GPU Acceleration (Primary)
The AI module prioritizes GPU execution using available backends:
- **CUDA**: For NVIDIA GPUs (RTX series, GTX 10-series and newer)
- **DirectML**: For AMD GPUs, Intel Arc, and modern integrated GPUs on Windows
- **Vulkan Compute**: Cross-platform fallback for non-CUDA GPUs (future consideration)

#### CPU Fallback (Secondary)
When no compatible GPU is detected or VRAM is insufficient, inference falls back to CPU execution using optimized libraries (ONNX Runtime, OpenVINO). Performance degrades significantly but remains functional.

### Backend Detection and Selection
At startup, the system probes available compute devices:
1. Check for NVIDIA GPU with CUDA support
2. Check for DirectML-compatible GPU
3. Fall back to CPU if no GPU available

User can override automatic selection via configuration (e.g., force CPU for testing).

### No Cloud, No API Keys
**Explicit guarantee**: The AI module contains no network code, no API authentication, and no telemetry. All model weights are stored locally. Users retain full control over their data and processing.

---

## AI Responsibilities

### What AI MUST Do
1. **Spatial Super-Resolution Only**: Take a single low-resolution frame and produce a high-resolution frame at 2× or 4× scale.
2. **Preserve Anime Characteristics**: Maintain sharp line art, flat color regions, and smooth gradients without introducing photorealistic artifacts.
3. **Operate Deterministically**: Given the same input, always produce the same output (no randomness, no generative variation).
4. **Respect Memory Limits**: Process frames in tiles if VRAM is insufficient for full-frame inference.

### What AI MUST NEVER Do
1. **No Temporal Processing**: AI does not access previous or future frames. Single-frame operation only.
2. **No Content Generation**: AI refines existing content via interpolation and learned priors. It does not "imagine" details that radically diverge from the input.
3. **No Style Transfer**: AI does not change the artistic style (e.g., converting anime to photorealistic).
4. **No Real-Time Inference**: AI does not run during video playback. Separation from the real-time shader path is mandatory.

### Why Spatial-Only Super-Resolution
**Rationale**: Temporal AI models (e.g., video super-resolution with recurrent networks) require:
- Multi-frame buffering (high memory cost)
- Sequential processing (cannot parallelize across frames)
- Motion estimation or optical flow (computationally expensive)
- Risk of temporal artifacts (ghosting, flickering)

Spatial-only inference:
- Processes each frame independently (parallelizable for batch export)
- Simpler model architecture (faster inference)
- Predictable memory usage
- Easier to integrate with existing video processing pipelines

---

## Input / Output Contract

### AI Input Tensor Specification

#### Color Space
- **RGB linear** or **sRGB** (backend-specific preference)
- Shader pipeline outputs normalized RGB; AI module accepts this directly

#### Normalization
- Pixel values in range `[0.0, 1.0]`
- Some models may require `[-1.0, 1.0]` via simple transform: `(x * 2.0) - 1.0`

#### Precision
- **FP32 (preferred)**: Ensures numerical stability during inference
- **FP16 (optional)**: Supported for memory-constrained GPUs, requires model quantization validation

#### Tensor Shape
- **NCHW layout**: `(batch, channels, height, width)`
  - Batch typically = 1 (single frame)
  - Channels = 3 (RGB)
  - Height/Width = tile dimensions (see Tile-Based Inference section)

#### Dynamic Resolution Support
Models must accept arbitrary input dimensions (within reasonable limits, e.g., 64×64 minimum, 2048×2048 maximum per tile).

### AI Output Tensor Specification

#### Scale Factors
- **2× upscaling**: Output dimensions = Input dimensions × 2
- **4× upscaling**: Output dimensions = Input dimensions × 4

Models are scale-specific: a 2× model produces 2× output, a 4× model produces 4× output. Cascading (e.g., 2× → 2× = 4×) is supported but discouraged due to error accumulation.

#### Format and Precision
- Same color space as input (RGB)
- Same normalization range `[0.0, 1.0]`
- FP32 or FP16 (matches input precision)

#### Tensor Shape
- **NCHW layout**: `(batch, channels, height_out, width_out)`
  - `height_out = height_in * scale_factor`
  - `width_out = width_in * scale_factor`

### Contract Stability
The input/output specification is **model-agnostic**. Any anime SR model conforming to this contract can be swapped in without changing the inference pipeline. This allows:
- Model updates without code changes
- User-provided custom models (advanced feature)
- A/B testing different models

---

## Model Selection Strategy

### Suitable Model Families

#### Real-ESRGAN (Anime Variant)
- **Strengths**: Trained specifically on anime data, preserves line art, handles compression artifacts well
- **Weaknesses**: Can over-sharpen, occasional texture hallucination
- **Use Case**: General-purpose anime upscaling

#### SwinIR / SwinIR-Anime
- **Strengths**: Transformer-based, excellent at preserving structure, lower hallucination risk
- **Weaknesses**: Slower inference than CNN-based models, larger model size
- **Use Case**: High-quality offline export

#### Compact Anime SR Models
- **Strengths**: Lightweight (e.g., <10MB), fast inference, low VRAM usage
- **Weaknesses**: Lower quality than heavy models, may blur fine details
- **Use Case**: CPU fallback, batch processing on lower-end hardware

### Selection Criteria
Models are chosen based on:
1. **Training Data**: Must be trained primarily on anime content, not natural images
2. **Artifact Control**: Should minimize ringing, halos, and texture noise
3. **Inference Speed**: Target <500ms per 1080p frame on mid-range GPUs (RTX 3060)
4. **Model Size**: Prefer <100MB for fast loading and distribution

### Risks and Mitigation

#### Hallucination
AI models may invent fine details not present in the source. For anime, this manifests as:
- Spurious texture in flat regions
- Artificial line art artifacts
- Over-sharpened edges

**Mitigation**: Use models trained with perceptual loss tuned for anime. Optionally blend AI output with traditional bicubic upscaling (e.g., 80% AI + 20% bicubic) to reduce hallucination.

#### Over-Sharpening
Models trained to maximize PSNR/SSIM metrics may produce unnaturally sharp results.

**Mitigation**: Post-process AI output with subtle Gaussian blur (sigma < 0.5) if sharpening is excessive. Allow user configuration of sharpening intensity.

### Explicitly Excluded Models

#### Diffusion Models
- **Why Excluded**: Stochastic (non-deterministic), extremely slow (seconds per frame), risk of style deviation
- **Examples**: Stable Diffusion upscalers, latent diffusion SR

#### Generic Real-World SR Models
- **Why Excluded**: Trained on natural images, produce photorealistic artifacts on anime
- **Examples**: ESRGAN (non-anime variant), EDSR trained on DIV2K

#### Temporal SR Models
- **Why Excluded**: Require multi-frame buffering, violate spatial-only constraint
- **Examples**: BasicVSR, RVRT

---

## Tile-Based Inference Design

### Why Tile-Based Inference is Mandatory

#### VRAM Constraints
A single 1080p frame upscaled to 4K at FP32 requires:
- Input: `1920 × 1080 × 3 × 4 bytes ≈ 25 MB`
- Output: `3840 × 2160 × 3 × 4 bytes ≈ 100 MB`
- Intermediate activations: Can exceed 1-2 GB depending on model depth

For 4× upscaling from 1080p, total VRAM usage may exceed 4 GB, which is problematic for 6 GB or 8 GB GPUs.

#### Solution
Process the frame in smaller tiles (e.g., 512×512), upscale each tile independently, and stitch results together. This keeps peak VRAM usage predictable regardless of input resolution.

### Tile Size Selection

#### Considerations
- **Larger tiles**: Better context for the model, fewer seam artifacts, but higher VRAM usage
- **Smaller tiles**: Lower VRAM, more parallelizable, but potential seam artifacts and loss of global context

#### Proposed Tile Sizes
- **512×512**: Default for most GPUs (2-4 GB VRAM)
- **768×768**: For high-end GPUs (8-12 GB VRAM)
- **256×256**: Fallback for low-VRAM or CPU inference

Tile size is dynamically selected based on available VRAM and model architecture.

### Overlap and Seam Prevention

#### Overlap Strategy
Process tiles with overlap (e.g., 32-64 pixels on each edge). The center region of each tile is kept, and overlapping edges are discarded or blended.

#### Blending Strategy
Use **feathered blending** in overlap regions:
- Linearly interpolate between adjacent tiles in the overlap zone
- Smooth transition prevents visible seams

Example: For 64-pixel overlap, the left tile contributes 100% at its center, fading to 0% at the overlap boundary. The right tile contributes 0% at the left overlap boundary, fading to 100% at its center.

#### Edge Cases
- **Image boundaries**: No overlap at frame edges; tiles extend to the border
- **Non-divisible dimensions**: Pad input to tile-friendly dimensions, crop output to original aspect ratio

### Memory and VRAM Management
- **Pre-allocate buffers**: Reuse input/output tensors across tiles to avoid fragmentation
- **Batch processing tiles**: Process multiple tiles in a batch if VRAM allows (e.g., batch=4 for 512×512 tiles on 12 GB GPU)
- **Streaming**: For very large images, stream tiles from disk rather than loading the entire image into RAM

---

## Inference Backend Abstraction

### Backend-Agnostic Interface
The AI module exposes a uniform interface to the rest of KizunaSR:

```
Interface: AIUpscaler
  - load_model(path: string, device: enum[CUDA, DirectML, CPU])
  - infer(input_tensor: Tensor, scale: int) -> Tensor
  - get_model_info() -> ModelMetadata
  - release_resources()
```

This abstraction allows the core pipeline to remain ignorant of which backend is active.

### Backend Implementations

#### CUDA Backend
- Uses **ONNX Runtime with CUDA Execution Provider** or **TensorRT** for optimized inference
- Models stored as `.onnx` or `.engine` files
- Leverages CUDA graphs for low-latency repeated inference

#### DirectML Backend
- Uses **ONNX Runtime with DirectML Execution Provider**
- Compatible with any DirectX 12-capable GPU (NVIDIA, AMD, Intel)
- Slightly slower than native CUDA but cross-vendor

#### CPU Backend
- Uses **ONNX Runtime CPU** or **OpenVINO** (for Intel CPUs)
- Multi-threaded inference across CPU cores
- 10-50× slower than GPU depending on model and CPU

### Model Loading from Disk
Models are stored locally in a designated directory (e.g., `models/anime_sr/`):
- `realESRGAN_anime_2x.onnx`
- `swinir_anime_4x.onnx`

At runtime:
1. User selects model via configuration
2. System loads model from disk into memory
3. Backend optimizes model for target device (e.g., TensorRT engine compilation)
4. Model remains in memory for subsequent frames (no reloading)

**No internet access required**. Users manually download models and place them in the model directory.

---

## Hybrid Pipeline Integration

### Shader Output → AI Input
When AI inference is enabled (export mode):
1. Shader pipeline processes frame through Normalize/Structural Reconstruction stages
2. Output is written to a GPU texture or CPU buffer
3. Texture is converted to tensor format (NCHW, FP32)
4. Tensor is passed to AI inference module

This allows AI to benefit from shader preprocessing (denoising, edge refinement) rather than operating on raw video.

### AI Output → Shader Post-Processing
After AI upscaling:
1. AI output tensor is converted back to GPU texture
2. Shader pipeline applies Perceptual Enhancement and Temporal Stabilization stages
3. Final output is encoded to video or displayed

This ensures consistency: both real-time and AI-enhanced outputs undergo the same post-processing, maintaining visual coherence.

### Real-Time Preview Consistency
When previewing AI-enhanced content:
- Real-time playback uses shader-only pipeline (fast)
- User can pause and trigger AI upscaling on current frame (slow, one-time)
- Paused frame displays AI result; playback resumes with shader pipeline

This "hybrid" mode allows users to preview AI quality without processing every frame.

---

## Performance Profiles

### Profile 1: Real-Time Safe (AI Disabled)
- **Use Case**: Live video playback at 60 fps
- **Pipeline**: Shader-only (Normalize → Structural Reconstruction → Real-Time Upscale → Perceptual Enhancement → Temporal Stabilization)
- **Performance**: <16ms per frame at 1080p → 4K
- **Quality**: Excellent for anime, good enough for most viewing

### Profile 2: Export / Quality Mode (AI Enabled)
- **Use Case**: Offline video export, frame upscaling for archival
- **Pipeline**: Shader preprocessing → AI Upscaling → Shader post-processing
- **Performance**: 0.5-2 seconds per frame depending on GPU
- **Quality**: State-of-the-art, maximum detail preservation

### Profile 3: Batch Processing Mode (AI Enabled, Optimized)
- **Use Case**: Upscaling entire video files (thousands of frames)
- **Pipeline**: Same as Export mode, but with optimizations:
  - Multi-tile batching
  - Frame queue (GPU never idle)
  - Parallel encoding while next frame processes
- **Performance**: 30-60 frames per minute on RTX 3060
- **Quality**: Identical to Export mode

### Automatic Fallback Strategy
If system detects insufficient resources:
- **Low VRAM**: Reduce tile size or fall back to CPU
- **Slow inference**: Warn user of long processing time, offer to cancel
- **Model load failure**: Fall back to shader-only pipeline with user notification

---

## Explicit Non-Goals

The AI inference module deliberately **DOES NOT** attempt the following:

### No Cloud AI
- No SaaS APIs (e.g., Topaz, waifu2x web services)
- No remote inference servers
- No telemetry or usage tracking
- **Rationale**: Privacy, cost, and offline-first design

### No API Usage or Authentication
- No API keys required
- No account creation or login
- No dependency on external services
- **Rationale**: User control and zero operational cost

### No Temporal AI
- No multi-frame super-resolution
- No optical flow or motion estimation
- No recurrent networks or hidden states
- **Rationale**: Complexity, memory overhead, and sequential processing bottleneck

### No Diffusion Models
- No denoising diffusion probabilistic models (DDPM)
- No latent diffusion super-resolution
- No stochastic inference
- **Rationale**: Non-deterministic output, extremely slow inference (10-60 seconds per frame), style instability

### No Live Inference During Playback
- AI does not run concurrently with video playback
- Real-time path remains shader-only
- **Rationale**: GPU contention would cause frame drops and stuttering. Separation ensures playback smoothness.

### No In-Place Model Training
- System does not fine-tune or retrain models
- Users cannot "teach" the AI with custom data
- **Rationale**: Training requires different infrastructure (PyTorch, large datasets, GPUs for hours). KizunaSR is an inference-only system.

---

## Implementation Boundaries

### What This Design Enables
- Drop-in model replacement (swap ONNX files)
- Multi-backend support (CUDA, DirectML, CPU)
- Tile-based processing for arbitrary resolutions
- Hybrid shader + AI pipeline
- Offline, privacy-preserving operation

### What Requires Future Work
- Model zoo management (model discovery, updates)
- Fine-grained user controls (hallucination reduction sliders)
- GPU memory profiling and auto-tuning
- Multi-GPU support for batch processing

---

## Conclusion

This specification defines a robust, local-first AI inference subsystem for KizunaSR. The design prioritizes user privacy, hardware compatibility, and clean integration with the existing shader pipeline. By maintaining strict separation between real-time and offline processing, KizunaSR can offer both fast playback and state-of-the-art export quality.

The tile-based architecture and backend abstraction ensure scalability across hardware configurations, from high-end RTX 4090 systems to modest integrated GPUs with CPU fallback.

---
*Document Version: 1.0*  
*Status: Design Complete - Ready for Implementation*

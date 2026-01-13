# Real-Time Shader Pipeline Specification

## Document Purpose
This specification defines the conceptual architecture of the real-time shader pipeline for KizunaSR. It describes the processing stages, their purposes, and how they work together to provide high-quality anime upscaling within real-time performance constraints.

---

## Pipeline Overview

The real-time pipeline consists of five sequential stages:

1. **Normalize** — Prepare input for consistent processing
2. **Structural Reconstruction** — Detect and refine anime-specific features
3. **Real-Time Upscale** — Increase resolution intelligently
4. **Perceptual Enhancement** — Refine visual quality
5. **Temporal Stabilization** — Reduce flicker and shimmer

Each stage is designed to be GPU-friendly, stateless (except for temporal), and optimized for anime content characteristics.

---

## Stage 1: Normalize

### Purpose
Convert input video frames into a consistent color space and dynamic range suitable for downstream processing. Remove encoding artifacts and standardize bit depth.

### Input Characteristics
- Arbitrary color spaces (BT.709, BT.601, etc.)
- Variable bit depths (8-bit, 10-bit)
- Potential compression artifacts from video codecs
- HDR or SDR content

### Output Characteristics
- Linear RGB color space or perceptually uniform space
- Normalized intensity range (0.0 to 1.0)
- Standardized precision (likely FP16)
- Clean baseline for edge detection

### Core Algorithmic Idea
Apply gamma correction or transfer function conversion to linearize pixel values. Optionally apply light denoising to remove compression artifacts without blurring intentional details. Map all inputs to a common working color space.

### Why Critical for Anime
Anime often has flat color regions with sharp boundaries. Compression artifacts can introduce false edges. Normalizing ensures that edge detection in later stages responds to actual line art, not encoding noise.

### Why Safe for Real-Time
- Per-pixel operation with no spatial dependencies
- Simple mathematical transforms (power functions, look-up tables)
- Minimal memory reads (single texture sample per pixel)
- No branching or complex logic

---

## Stage 2: Structural Reconstruction

### Purpose
Identify and enhance anime-specific structural elements: line art, flat color regions, gradients, and textures. Prepare edge maps and feature masks for intelligent upscaling.

### Input Characteristics
- Normalized, clean frame data
- Consistent color representation
- Potential aliasing on edges from source resolution

### Output Characteristics
- Edge strength maps (where line art exists)
- Feature classification (flat vs gradient vs texture)
- Refined anti-aliasing on detected edges
- Preserved sharp boundaries

### Core Algorithmic Idea
Use directional gradient analysis to detect edges. Classify pixels based on local variance: flat regions have near-zero variance, gradients have smooth directional change, textures have high-frequency variation. Apply edge-aware smoothing to reduce aliasing while preserving intentional sharpness.

### Why Critical for Anime
Anime is structurally distinct from photographic content:
- **Line art**: High-contrast black or colored outlines
- **Cel shading**: Large uniform color regions
- **Intentional gradients**: Smooth color transitions for depth
- **Minimal texture detail**: Unlike real-world imagery

Treating anime like natural images causes edge ringing and loss of flat region purity. Structural reconstruction adapts processing to these characteristics.

### Why Safe for Real-Time
- Local operations only (typically 3x3 or 5x5 neighborhoods)
- No iterative solvers or global optimization
- Edge detection is embarrassingly parallel
- Feature classification uses simple thresholds

---

## Stage 3: Real-Time Upscale

### Purpose
Increase spatial resolution from input to target size using edge-aware interpolation guided by the structural information from Stage 2.

### Input Characteristics
- Normalized color data
- Edge maps and feature masks
- Source resolution (e.g., 1080p)

### Output Characteristics
- Target resolution (e.g., 4K)
- Sharp edges along line art
- Smooth gradients without staircasing
- No ringing artifacts in flat regions

### Core Algorithmic Idea
Use directional interpolation weighted by edge orientation. Along detected edges, sample pixels parallel to the edge direction to avoid blurring. In flat regions, use simple bilinear or bicubic sampling. In gradient regions, use smooth upsampling kernels. Essentially: content-adaptive filtering.

### Why Critical for Anime
Standard bilinear or bicubic upscaling blurs edges, making line art lose definition. Anime viewers are highly sensitive to blurry lines. Edge-directed upscaling maintains crispness while avoiding the artificial sharpening halos common in traditional sharpening filters.

### Why Safe for Real-Time
- Still a local operation (kernel-based)
- Directional filters are slightly more expensive than bilinear but still simple
- No global information required
- Texture sampling is hardware-accelerated
- Modern GPUs handle this pattern efficiently

---

## Stage 4: Perceptual Enhancement

### Purpose
Apply finishing touches to improve subjective visual quality: contrast refinement, line art darkening, color vibrancy adjustment, and subtle sharpening.

### Input Characteristics
- Upscaled frame at target resolution
- Edge maps (still available from Stage 2)
- Feature classifications

### Output Characteristics
- Visually appealing output with "pop"
- Enhanced line art definition
- Balanced contrast without clipping
- Vibrant colors without oversaturation

### Core Algorithmic Idea
Apply selective enhancements based on pixel classification. Darken line art slightly to increase perceived sharpness. Enhance local contrast in gradient regions using unsharp masking or subtle S-curve adjustments. Optionally boost color saturation in flat regions. All adjustments are conservative to avoid artifacts.

### Why Critical for Anime
Anime benefits from "punchy" visuals. Line art can appear washed out after upscaling, especially if anti-aliased. Subtle enhancements restore the visual impact without looking over-processed. This stage compensates for the inherent softness introduced by any upscaling process.

### Why Safe for Real-Time
- Per-pixel or small-kernel operations
- Enhancements are applied with simple multiplications and additions
- No expensive operations like global histogram analysis
- All processing is local and parallel

---

## Stage 5: Temporal Stabilization

### Purpose
Reduce frame-to-frame flicker and shimmer caused by slight changes in edge detection or interpolation between frames. Provide temporal coherence without introducing motion blur or ghosting.

### Input Characteristics
- Current frame (fully processed through Stages 1-4)
- Previous frame from history buffer
- No motion vectors (too expensive for real-time)

### Output Characteristics
- Temporally stable edges
- Reduced flickering in flat regions
- No visible ghosting during motion
- Clean frame ready for display

### Core Algorithmic Idea
Blend the current frame with the previous frame using a small temporal weight (e.g., 10-20% contribution from history). Use pixel difference as a motion detector: if current and previous pixels differ significantly, assume motion and skip blending to avoid ghosting. If pixels are similar, assume static content and apply temporal smoothing.

### Why Critical for Anime
Anime often has static backgrounds with moving characters. Edge detection can produce slight sub-pixel variations frame-to-frame, causing visible shimmer. Temporal stabilization smooths these variations in static regions while preserving motion sharpness.

### Why Safe for Real-Time
- Requires only one previous frame (minimal memory overhead)
- Difference-based motion detection is trivial to compute
- Blending is a simple weighted average
- No optical flow or motion estimation required
- History buffer is small (one full-resolution frame)

---

## Stage Chaining and Execution Order

### Execution Sequence
The stages execute in strict sequential order:

```
Input Frame → Normalize → Structural Reconstruction → Real-Time Upscale → 
Perceptual Enhancement → Temporal Stabilization → Output Frame
```

### Why This Order

1. **Normalize first**: All downstream stages assume consistent color space and dynamic range. Normalization must happen before any analysis.

2. **Structural Reconstruction before Upscale**: Edge maps must be generated at source resolution before upscaling. Detecting edges after upscaling would find interpolation artifacts instead of true features.

3. **Upscale in the middle**: This is the resolution-changing stage. Processing before upscaling operates on smaller data (faster). Processing after upscaling operates on full resolution (more detail available for refinement).

4. **Perceptual Enhancement after Upscale**: Enhancements like sharpening are best applied at target resolution to avoid amplifying upscaling artifacts. Operating on the final resolution allows precise control of the output appearance.

5. **Temporal Stabilization last**: Must operate on fully processed frames to smooth the final output. Applying it earlier would stabilize intermediate results but still allow flicker in later stages.

### What Breaks If Order Changes

- **Upscaling before edge detection**: Interpolation creates false edges, wasting computation and producing inferior results.
- **Enhancement before upscaling**: Enhancements get resampled and diluted during upscaling.
- **Temporal before enhancement**: Enhancements still flicker frame-to-frame.
- **Normalization not first**: Inconsistent inputs cause unpredictable edge detection and color processing.

---

## Temporal Stabilization Details

### History Buffer Usage
Maintain a single frame buffer containing the previous output frame. Each new frame reads from this buffer, blends with the current result, and writes back to replace it. This circular pattern requires minimal memory: exactly one frame at output resolution.

### Shimmer Reduction Strategy
Shimmer occurs when:
- Sub-pixel edge positions oscillate
- Interpolation weights vary slightly
- Floating-point precision causes tiny color shifts

**Solution**: For each pixel, compute the absolute difference between current and previous values. If difference is below a threshold (suggesting static content), blend with a small temporal weight (e.g., 15%). If difference is above threshold (suggesting motion), use 100% current frame.

This adaptive approach:
- Smooths static regions (no perceptible blur)
- Preserves moving content (no ghosting)
- Requires no motion vectors
- Self-adjusts to content characteristics

### No Motion Vector Requirement
Motion vectors are expensive to compute and require multi-frame analysis or optical flow algorithms. For real-time anime processing, simple difference-based detection suffices because:
- Anime has distinct moving objects on static backgrounds
- Large differences clearly indicate motion
- False positives (treating static as motion) are safe—they just skip stabilization
- False negatives (treating motion as static) are rare due to significant color changes during movement

---

## Performance Constraints

### Real-Time Definition
A shader is considered "acceptable for real-time" if it can process a 1080p frame in under 16 milliseconds on mid-range GPUs (e.g., NVIDIA RTX 3060, AMD RX 6600). This allows 60 fps video playback with minimal latency.

For 4K output, targeting 30-60 fps is acceptable, relaxing the constraint to 16-33ms.

### Instruction Count Philosophy
Each stage should minimize:
- **Branching**: Conditional logic hurts GPU parallelism. Use mathematics instead (e.g., smooth step functions instead of if statements).
- **Texture samples**: Limit to 4-9 samples per pixel in most stages. Structural reconstruction may sample 9 pixels (3x3 neighborhood).
- **Compute intensity**: Prefer simple arithmetic. Avoid expensive functions like sine, cosine, logarithm unless absolutely necessary.

Rough guideline: Each stage should complete in 2-4ms on target hardware for 1080p.

### Memory Bandwidth Considerations
Modern GPUs are bandwidth-limited. Optimization strategies:
- **Minimize texture reads**: Reuse sampled values across computations.
- **Use FP16 where possible**: Half precision reduces bandwidth by 2x and increases throughput on modern GPUs.
- **Avoid redundant writes**: Write to intermediate buffers only when multiple stages need the same data.
- **Coalesce memory access**: Ensure shader memory access patterns are cache-friendly (sequential, aligned).

### Precision Choices
- **FP16 (half precision)**: Suitable for color values, edge strengths, blend weights. Sufficient dynamic range for visual content.
- **FP32 (single precision)**: Only when necessary for accumulation buffers or when precision loss would be visible (rare).
- **Integer precision**: Not typically useful for this pipeline (color processing requires fractional values).

Strategy: Use FP16 aggressively, validate visually, upgrade to FP32 only if artifacts appear.

---

## Explicit Non-Goals

The real-time shader pipeline deliberately **DOES NOT** attempt the following:

### No AI Inference
- No neural networks
- No learned upscaling models (e.g., no ESRGAN, no Real-ESRGAN)
- No pretrained weights loaded into shaders

**Rationale**: AI inference is handled by a separate pipeline path in KizunaSR. The shader pipeline focuses purely on algorithmic, heuristic-based processing.

### No Texture Hallucination
- No generation of fine details that don't exist in the source
- No synthesis of high-frequency content beyond interpolation
- No "imagining" textures or patterns

**Rationale**: Real-time shaders lack the computational budget for plausible hallucination. Attempting it produces unconvincing artifacts. Upscaling is interpolation-based, not generative.

### No Heavy Temporal Logic
- No multi-frame motion estimation
- No optical flow computation
- No temporal super-resolution (combining multiple frames)
- No complex motion-compensated filtering

**Rationale**: These techniques require significant computation and memory. The lightweight temporal stabilization (single-frame blending) provides sufficient flicker reduction for real-time use.

### No Offline-Quality Reconstruction
- Cannot match the quality of multi-second-per-frame AI upscalers
- Cannot perfectly reconstruct line art from heavily compressed sources
- Cannot remove all artifacts from low-bitrate video

**Rationale**: Real-time processing is a time-quality trade-off. The shader pipeline provides excellent results for clean sources but is not a substitute for offline, heavy-compute processing when maximum quality is required.

---

## Conclusion

This specification defines a practical, implementable real-time shader pipeline optimized for anime content. Each stage has clear responsibilities, and the overall design respects GPU performance constraints while delivering visually pleasing results.

The pipeline is modular: individual stages can be refined or replaced without disrupting the overall architecture. Future work may adjust parameters, improve algorithms within each stage, or add optional branches (e.g., different upscaling kernels), but the five-stage structure provides a solid foundation.

---
*Document Version: 1.0*  
*Status: Design Complete - Ready for Implementation*

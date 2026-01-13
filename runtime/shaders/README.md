# KizunaSR Real-Time Shader Pipeline - Implementation Guide

## Overview
This directory contains the minimal viable implementation of the 5-stage real-time shader pipeline for anime upscaling, based on the frozen design specification.

## Shader Files

### Shared Vertex Shader
- **`fullscreen.vert`**: Simple vertex shader for fullscreen quad rendering
  - Used by all pipeline stages
  - Maps NDC coordinates (-1 to 1) to texture coordinates (0 to 1)

### Fragment Shaders (Pipeline Stages)

#### Stage 1: Normalize (`01_normalize.frag`)
**Purpose**: Convert input to consistent linear RGB color space and remove compression artifacts

**Implementation**:
- sRGB to linear RGB conversion using accurate gamma correction
- Lightweight bilateral-style denoising (3×3 neighborhood)
- Color similarity-based weighting to preserve edges

**Input**: Raw frame texture (sRGB or arbitrary color space)  
**Output**: Normalized linear RGB in [0.0, 1.0] range

**Parameters**:
- `GAMMA = 2.2`: Gamma value for sRGB conversion
- `DENOISE_STRENGTH = 0.05`: Subtle denoising strength

---

#### Stage 2: Structural Reconstruction (`02_structural_reconstruction.frag`)
**Purpose**: Detect edges and classify regions for intelligent upscaling

**Implementation**:
- Sobel edge detection (3×3 kernel) on luminance
- Gradient magnitude calculation for edge strength
- Edge-aware smoothing (reduces aliasing on weak edges, preserves strong edges)

**Input**: Normalized linear RGB from Stage 1  
**Output**: RGB color + edge strength in alpha channel

**Parameters**:
- `EDGE_THRESHOLD = 0.1`: Threshold for edge normalization
- `SMOOTHING_STRENGTH = 0.3`: Edge-aware smoothing factor

---

#### Stage 3: Real-Time Upscale (`03_realtime_upscale.frag`)
**Purpose**: Increase resolution using edge-aware interpolation

**Implementation**:
- Bicubic interpolation for smooth regions
- Edge-directed sampling for strong edges (samples parallel to edge)
- Adaptive blending based on edge strength
- Atan-based edge direction detection

**Input**: Processed frame from Stage 2 (RGB + edge in alpha)  
**Output**: Upscaled frame at target resolution (typically 2× source)

**Uniforms**:
- `uSourceSize`: Source resolution (vec2)
- `uTargetSize`: Target resolution (vec2)
- `uSharpness`: Edge sharpening strength (float)

**Parameters**:
- `EDGE_THRESHOLD = 0.3`: Threshold for edge-directed mode activation

---

#### Stage 4: Perceptual Enhancement (`04_perceptual_enhancement.frag`)
**Purpose**: Enhance visual quality with contrast, sharpening, and saturation

**Implementation**:
- Unsharp masking for sharpening (3×3 Gaussian blur + detail boost)
- Line art darkening (targets dark pixels < 0.3 luminance)
- S-curve contrast adjustment
- HSV-based saturation boost

**Input**: Upscaled frame from Stage 3  
**Output**: Enhanced frame with improved visual appeal

**Uniforms**:
- `uContrastBoost`: Contrast multiplier (1.0 = no change, 1.2 = 20% boost)
- `uSaturationBoost`: Saturation multiplier (1.0 = no change, 1.1 = 10% boost)
- `uSharpening`: Sharpening strength (0.0 = none, 0.5 = moderate)
- `uLineDarkening`: Line darkening strength (0.0 = none, 0.2 = 20% darker)

---

#### Stage 5: Temporal Stabilization (`05_temporal_stabilization.frag`)
**Purpose**: Reduce flicker and shimmer using previous frame blending

**Implementation**:
- Motion detection via RGB difference (max channel difference)
- Adaptive blending with smoothstep for smooth transition
- Static regions: blend with history (reduce flicker)
- Moving regions: use current frame only (avoid ghosting)

**Input**: Enhanced frame from Stage 4 + previous output frame (history buffer)  
**Output**: Temporally stabilized final frame

**Uniforms**:
- `uTemporalWeight`: Blend weight for history (e.g., 0.15 = 15% history)
- `uMotionThreshold`: Pixel difference threshold for motion detection
- `uFirstFrame`: Boolean flag to skip temporal on first frame

---

## Pipeline Execution Model

### Resource Requirements

**Textures/Buffers**:
1. **Input texture**: Raw video frame
2. **Normalized texture**: Output of Stage 1 (FP16 or FP32, source resolution)
3. **Processed texture**: Output of Stage 2 (RGBA, source resolution, alpha = edge strength)
4. **Upscaled texture**: Output of Stage 3 (RGB, target resolution)
5. **Enhanced texture**: Output of Stage 4 (RGB, target resolution)
6. **History buffer**: Previous output frame (RGB, target resolution) - persistent
7. **Output texture**: Final result from Stage 5

**Memory Estimate** (1080p → 4K):
- Source textures (1920×1080): ~8 MB each × 3 stages = 24 MB
- Target textures (3840×2160): ~32 MB each × 3 stages = 96 MB
- History buffer: 32 MB
- **Total**: ~150 MB VRAM

### Per-Frame Execution Sequence

```
Frame N arrives:
├─ Bind Stage 1 (Normalize)
│  ├─ Input: Raw frame texture
│  ├─ Output: Normalized texture
│  └─ Execute fullscreen quad
│
├─ Bind Stage 2 (Structural Reconstruction)
│  ├─ Input: Normalized texture
│  ├─ Output: Processed texture
│  └─ Execute fullscreen quad
│
├─ Bind Stage 3 (Real-Time Upscale)
│  ├─ Input: Processed texture
│  ├─ Output: Upscaled texture
│  ├─ Set uniforms: uSourceSize, uTargetSize, uSharpness
│  └─ Execute fullscreen quad at target resolution
│
├─ Bind Stage 4 (Perceptual Enhancement)
│  ├─ Input: Upscaled texture
│  ├─ Output: Enhanced texture
│  ├─ Set uniforms: uContrastBoost, uSaturationBoost, etc.
│  └─ Execute fullscreen quad
│
└─ Bind Stage 5 (Temporal Stabilization)
   ├─ Input: Enhanced texture, History buffer
   ├─ Output: Final output texture
   ├─ Set uniforms: uTemporalWeight, uMotionThreshold, uFirstFrame
   ├─ Execute fullscreen quad
   └─ Copy output to history buffer for next frame
```

### Synchronization

- All stages execute sequentially (no parallelism between stages)
- Each stage must complete before the next begins
- GPU pipeline barriers between stages ensure data availability
- No CPU synchronization required (all GPU-resident)

---

## Validation and Testing

### Visual Validation

**Expected Results**:
1. **Stage 1**: Should remove compression blocking, smooth noise, appear slightly softer
2. **Stage 2**: Edge maps should highlight line art clearly (visible in alpha channel)
3. **Stage 3**: Upscaled output should have sharp edges on line art, smooth elsewhere
4. **Stage 4**: Enhanced output should have punchy colors, darker lines, higher contrast
5. **Stage 5**: Temporal output should reduce shimmer in static regions, no ghosting in motion

**Common Artifacts to Check**:
- **Ringing/halos** around edges → Reduce sharpening or edge-directed strength
- **Blocked/banded** flat regions → Check normalization and denoising
- **Ghosting trails** during motion → Reduce temporal weight or increase motion threshold
- **Flickering** in static regions → Increase temporal weight
- **Over-saturation** → Reduce saturation boost
- **Crushed blacks** → Reduce line darkening

### Performance Validation

**Target**: <16ms total pipeline time for 1080p → 4K at 60fps

**Profiling Per-Stage** (Estimated on RTX 3060):
1. Normalize: ~1-2ms
2. Structural Reconstruction: ~2-3ms (Sobel is cheap)
3. Real-Time Upscale: ~4-6ms (bicubic + edge sampling)
4. Perceptual Enhancement: ~2-3ms
5. Temporal Stabilization: ~1-2ms

**Total**: ~10-16ms (within budget)

**If Too Slow**:
- Reduce upscale factor (2× instead of 4×)
- Simplify bicubic to bilinear in Stage 3
- Reduce neighborhood size in denoising (1px radius instead of 1.5px)
- Disable temporal stabilization
- Use FP16 textures instead of FP32

### Correctness Validation

**Test Cases**:
1. **Solid color frame**: Should remain unchanged (no artifacts)
2. **Sharp line art**: Edges should be crisp, no blurring
3. **Smooth gradient**: No banding or staircasing
4. **Static background + moving character**: Background stable, character sharp
5. **High-compression video**: Denoising should reduce blocking without over-blurring

**Automated Checks**:
- Ensure output range is [0.0, 1.0] (no clipping or underflow)
- Verify texture dimensions match expectations at each stage
- Check for NaN or Inf values in output (shader validation layers)

---

## Integration Notes

### Required Uniform Updates Per Frame

**Stage 3 (Upscale)**:
- `uSourceSize`: Set to input resolution
- `uTargetSize`: Set to output resolution
- `uSharpness`: User-configurable (default 0.7)

**Stage 4 (Enhancement)**:
- `uContrastBoost`: User-configurable (default 1.1)
- `uSaturationBoost`: User-configurable (default 1.05)
- `uSharpening`: User-configurable (default 0.3)
- `uLineDarkening`: User-configurable (default 0.15)

**Stage 5 (Temporal)**:
- `uTemporalWeight`: User-configurable (default 0.15)
- `uMotionThreshold`: Content-dependent (default 0.1 for anime)
- `uFirstFrame`: Set to true on first frame, false thereafter

### Graphics API Integration

**Vulkan**:
- Compile shaders to SPIR-V using `glslangValidator` or `glslc`
- Create pipeline for each stage with fullscreen.vert + stage.frag
- Use framebuffer objects for intermediate textures
- Insert pipeline barriers between stages

**OpenGL**:
- Compile shaders at runtime
- Create framebuffer objects for intermediate textures
- Bind textures to appropriate texture units
- Use `glMemoryBarrier` if needed (likely not required for sequential execution)

**DirectX**:
- Compile shaders to DXBC/DXIL
- Use render target views for intermediate textures
- Resource barriers between stages

---

## Limitations and Future Work

### Current Limitations
- Fixed 2× upscale factor (requires uniform adjustment for 4×)
- No multi-level edge detection (only single threshold)
- Simple bicubic kernel (could be improved with Lanczos or Mitchell-Netravali)
- No GPU-specific optimizations (e.g., compute shaders, subgroup operations)

### Future Improvements
- Adaptive upscale factor (1.5×, 2×, 3×, 4×)
- Multi-scale edge detection for finer feature classification
- Higher-quality resampling kernels
- Compute shader variants for better performance
- FP16 texture paths for memory bandwidth reduction
- Tile-based processing for large resolutions (8K+)

---

## Conclusion

This MVP implementation provides a working, correct real-time shader pipeline that adheres to the frozen design specification. It prioritizes clarity and correctness over ultimate performance or quality, serving as a solid foundation for future optimization and enhancement.

All shaders are GPU-safe (no dynamic loops, limited texture samples, bounded execution time) and should run comfortably within the 16ms budget on mid-range GPUs for 1080p → 4K upscaling at 60fps.

---
*Implementation Version: 1.0 MVP*  
*Status: Complete - Ready for Integration and Testing*

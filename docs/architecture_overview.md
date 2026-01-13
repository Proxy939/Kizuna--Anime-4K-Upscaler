# Architecture Overview

## Purpose
High-level structural diagram and component breakdown of the KizunaSR system.

## System Components

### 1. Real-Time Shader Pipeline
See [Real-Time Shader Pipeline Design](realtime_shader_pipeline.md) for complete specification.

**Five Processing Stages**:
- Normalize → Structural Reconstruction → Real-Time Upscale → Perceptual Enhancement → Temporal Stabilization

**Performance**: <16ms per frame at 1080p → 4K on mid-range GPUs

### 2. Local AI Inference Module
See [AI Inference Module Design](ai_inference_module.md) for complete specification.

**Key Features**:
- Local-first execution (CUDA, DirectML, CPU backends)
- Tile-based processing for memory efficiency
- Model-agnostic tensor interface
- Support for Real-ESRGAN, SwinIR, and compact anime SR models

### 3. Hybrid Integration Layer
See [Hybrid Pipeline Integration](hybrid_pipeline_integration.md) for complete specification.

**Responsibilities**:
- Data flow management (shader → AI → shader)
- Preview vs export path separation
- Failure handling and graceful fallback
- Tile extraction and reassembly
- Export pipeline orchestration

## Architectural Principles

### Separation of Concerns
- **Core**: Pipeline contracts and interfaces
- **Runtime**: Execution backends (shader, AI)
- **Profiles**: Configuration-driven behavior

### Performance Isolation
- Real-time playback: Shader-only (never blocked by AI)
- Export mode: Shader + AI hybrid (quality-optimized)

### Graceful Degradation
- CUDA → DirectML → CPU → Shader-only fallback chain
- No crashes on AI failures

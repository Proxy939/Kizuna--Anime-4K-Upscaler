# Pipeline Specification

## Purpose
Defines the data flow and execution stages for the KizunaSR upscaling pipeline.

## Real-Time Shader Pipeline
For the complete specification of the real-time shader pipeline, see:
- **[Real-Time Shader Pipeline Design](realtime_shader_pipeline.md)**

This document covers:
- Five processing stages (Normalize, Structural Reconstruction, Real-Time Upscale, Perceptual Enhancement, Temporal Stabilization)
- Stage execution order and dependencies
- Performance constraints and optimization strategies
- Temporal stabilization approach
- Explicit non-goals

## AI Pipeline
For the complete specification of the local AI inference module, see:
- **[AI Inference Module Design](ai_inference_module.md)**

This document covers:
- Local-first execution model (CUDA, DirectML, CPU backends)
- Input/output tensor contracts
- Model selection strategy (Real-ESRGAN, SwinIR, compact models)
- Tile-based inference for memory efficiency
- Backend abstraction layer
- Hybrid shader + AI integration
- Performance profiles (real-time vs export modes)
- Explicit non-goals (no cloud, no temporal AI, no diffusion)

"""
KizunaSR - Hybrid Pipeline Orchestrator
========================================
Integrates shader stages (CPU reference) with AI inference module.

Supports two execution modes:
1. Preview Mode: Shader-only (fast, no AI)
2. Export Mode: Shader → AI Upscale → Shader (quality, slow)
"""

import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
from pathlib import Path

from core.shader_stages import (
    stage_normalize,
    stage_structural_reconstruction,
    stage_realtime_upscale,
    stage_perceptual_enhancement,
    stage_temporal_stabilization,
    linear_to_srgb
)
from runtime.ai.ai_inference import AIInferenceEngine


class PipelineConfig:
    """Configuration for KizunaSR pipeline."""
    
    def __init__(self):
        # Mode selection
        self.use_ai: bool = False  # False = preview, True = export with AI
        
        # Shader upscale factor (preview mode only)
        self.shader_scale: int = 2
        
        # AI configuration (export mode only)
        self.ai_model_path: Optional[str] = None
        self.ai_tile_size: int = 512
        self.ai_tile_overlap: int = 64
        self.ai_scale: int = 2
        
        # Enhancement parameters
        self.contrast_boost: float = 1.1
        self.saturation_boost: float = 1.05
        self.sharpening: float = 0.3
        self.line_darkening: float = 0.15
        
        # Temporal stabilization
        self.enable_temporal: bool = False
        self.temporal_weight: float = 0.15
        self.motion_threshold: float = 0.1
    
    def as_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            'use_ai': self.use_ai,
            'shader_scale': self.shader_scale,
            'ai_model_path': self.ai_model_path,
            'ai_tile_size': self.ai_tile_size,
            'ai_tile_overlap': self.ai_tile_overlap,
            'ai_scale': self.ai_scale,
            'contrast_boost': self.contrast_boost,
            'saturation_boost': self.saturation_boost,
            'sharpening': self.sharpening,
            'line_darkening': self.line_darkening,
            'enable_temporal': self.enable_temporal,
            'temporal_weight': self.temporal_weight,
            'motion_threshold': self.motion_threshold,
        }


class KizunaSRPipeline:
    """
    Hybrid KizunaSR pipeline orchestrator.
    
    Executes shader stages and optionally integrates AI upscaling.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.ai_engine: Optional[AIInferenceEngine] = None
        self.history_buffer: Optional[np.ndarray] = None
        
        # Initialize AI if enabled
        if config.use_ai:
            if config.ai_model_path is None:
                raise ValueError("AI mode enabled but no model path provided")
            
            self.ai_engine = AIInferenceEngine(
                model_path=config.ai_model_path,
                tile_size=config.ai_tile_size,
                tile_overlap=config.ai_tile_overlap,
                scale_factor=config.ai_scale
            )
            print(f"[Pipeline] AI upscaling enabled ({config.ai_scale}×)")
        else:
            print(f"[Pipeline] Preview mode (shader-only {config.shader_scale}×)")
    
    def process_frame(self, input_image: Image.Image) -> Image.Image:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            input_image: PIL RGB image
        
        Returns:
            Processed PIL RGB image
        """
        # Convert to numpy
        img_np = np.array(input_image)
        
        print(f"[Pipeline] Processing frame: {input_image.width}×{input_image.height}")
        
        # ======================================================================
        # Shader Preprocessing
        # ======================================================================
        
        # Stage 1: Normalize (sRGB → linear, denoise)
        print("[Pipeline] Stage 1: Normalize")
        normalized = stage_normalize(img_np)
        
        # Stage 2: Structural Reconstruction (edge detection, smoothing)
        print("[Pipeline] Stage 2: Structural Reconstruction")
        processed, edge_map = stage_structural_reconstruction(normalized)
        
        # ======================================================================
        # Upscaling (Branch: AI or Shader)
        # ======================================================================
        
        if self.config.use_ai and self.ai_engine is not None:
            # Export mode: AI upscaling
            print(f"[Pipeline] Stage 3: AI Upscale ({self.config.ai_scale}×)")
            
            # Convert to PIL for AI engine
            intermediate_img = Image.fromarray(
                (processed * 255).astype(np.uint8), mode='RGB'
            )
            
            # AI upscaling
            upscaled_pil = self.ai_engine.upscale(intermediate_img)
            upscaled = np.array(upscaled_pil).astype(np.float32) / 255.0
            
        else:
            # Preview mode: Shader upscaling
            print(f"[Pipeline] Stage 3: Shader Upscale ({self.config.shader_scale}×)")
            upscaled = stage_realtime_upscale(
                processed,
                edge_map,
                scale=self.config.shader_scale
            )
        
        # ======================================================================
        # Shader Post-Processing
        # ======================================================================
        
        # Stage 4: Perceptual Enhancement
        print("[Pipeline] Stage 4: Perceptual Enhancement")
        enhanced = stage_perceptual_enhancement(
            upscaled,
            contrast_boost=self.config.contrast_boost,
            saturation_boost=self.config.saturation_boost,
            sharpening=self.config.sharpening,
            line_darkening=self.config.line_darkening
        )
        
        # Stage 5: Temporal Stabilization (optional)
        if self.config.enable_temporal:
            print("[Pipeline] Stage 5: Temporal Stabilization")
            enhanced = stage_temporal_stabilization(
                enhanced,
                self.history_buffer,
                temporal_weight=self.config.temporal_weight,
                motion_threshold=self.config.motion_threshold
            )
            # Update history buffer
            self.history_buffer = enhanced.copy()
        
        # ======================================================================
        # Output Conversion
        # ======================================================================
        
        # Convert back to sRGB uint8
        # (enhanced is already in linear RGB [0, 1], need to convert to sRGB)
        output_srgb = linear_to_srgb(enhanced)
        
        # Convert to PIL
        output_pil = Image.fromarray(output_srgb, mode='RGB')
        
        print(f"[Pipeline] Output: {output_pil.width}×{output_pil.height}")
        
        return output_pil
    
    def reset_temporal_state(self):
        """Reset temporal history buffer."""
        self.history_buffer = None
        print("[Pipeline] Temporal state reset")


# ==============================================================================
# Convenience Functions
# ==============================================================================

def process_single_image(
    input_path: str,
    output_path: str,
    config: PipelineConfig
):
    """
    Process a single image through the pipeline.
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        config: Pipeline configuration
    """
    # Load input
    input_img = Image.open(input_path).convert('RGB')
    
    # Create pipeline
    pipeline = KizunaSRPipeline(config)
    
    # Process
    output_img = pipeline.process_frame(input_img)
    
    # Save
    output_img.save(output_path)
    print(f"[Success] Saved output to: {output_path}")


def process_image_sequence(
    input_dir: str,
    output_dir: str,
    config: PipelineConfig,
    pattern: str = "*.png"
):
    """
    Process a sequence of images.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        config: Pipeline configuration
        pattern: Glob pattern for input files
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get sorted list of input files
    input_files = sorted(input_path.glob(pattern))
    
    if not input_files:
        print(f"[Error] No files found matching {pattern} in {input_dir}")
        return
    
    print(f"[Sequence] Processing {len(input_files)} frames")
    
    # Create pipeline
    pipeline = KizunaSRPipeline(config)
    
    # Process each frame
    for i, input_file in enumerate(input_files):
        print(f"\n[Frame {i+1}/{len(input_files)}] {input_file.name}")
        
        # Load input
        input_img = Image.open(input_file).convert('RGB')
        
        # Process
        output_img = pipeline.process_frame(input_img)
        
        # Save with same name
        output_file = output_path / input_file.name
        output_img.save(output_file)
    
    print(f"\n[Success] Processed {len(input_files)} frames")
    print(f"[Output] Saved to: {output_dir}")


def main():
    """Example usage."""
    
    # =========================================================================
    # Example 1: Preview Mode (Shader-Only)
    # =========================================================================
    
    print("=" * 60)
    print("Example 1: Preview Mode (Shader-Only)")
    print("=" * 60)
    
    preview_config = PipelineConfig()
    preview_config.use_ai = False
    preview_config.shader_scale = 2
    preview_config.enable_temporal = False
    
    try:
        process_single_image(
            input_path="test_input.png",
            output_path="test_output_preview.png",
            config=preview_config
        )
    except FileNotFoundError:
        print("[Info] test_input.png not found, skipping preview example")
    
    # =========================================================================
    # Example 2: Export Mode (AI + Shader)
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("Example 2: Export Mode (AI + Shader)")
    print("=" * 60)
    
    export_config = PipelineConfig()
    export_config.use_ai = True
    export_config.ai_model_path = "models/anime_sr_2x.onnx"
    export_config.ai_scale = 2
    export_config.enable_temporal = False
    
    try:
        process_single_image(
            input_path="test_input.png",
            output_path="test_output_export.png",
            config=export_config
        )
    except FileNotFoundError as e:
        print(f"[Info] {e}, skipping export example")


if __name__ == "__main__":
    main()

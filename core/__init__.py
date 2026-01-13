"""KizunaSR - Core Pipeline Module"""

from .pipeline import KizunaSRPipeline, PipelineConfig, process_single_image, process_image_sequence
from .shader_stages import (
    stage_normalize,
    stage_structural_reconstruction,
    stage_realtime_upscale,
    stage_perceptual_enhancement,
    stage_temporal_stabilization
)

__all__ = [
    'KizunaSRPipeline',
    'PipelineConfig',
    'process_single_image',
    'process_image_sequence',
    'stage_normalize',
    'stage_structural_reconstruction',
    'stage_realtime_upscale',
    'stage_perceptual_enhancement',
    'stage_temporal_stabilization',
]
__version__ = '1.0.0-mvp'

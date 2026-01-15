"""
Real-ESRGAN AI Inference Module
================================
Production-ready Real-ESRGAN integration with GPU/CPU support.
"""

import os
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from PIL import Image

try:
    import torch
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError as e:
    raise ImportError(
        "Real-ESRGAN dependencies not installed. "
        "Run: pip install realesrgan==0.3.0 basicsr==1.4.2 facexlib gfpgan"
    ) from e


class RealESRGANInference:
    """Real-ESRGAN inference engine with tile-based processing."""
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "realesrgan-animevideov3",
        scale: int = 4,
        tile_size: int = 256,
        tile_pad: int = 10,
        gpu_id: Optional[int] = None
    ):
        """
        Initialize Real-ESRGAN upscaler.
        
        Args:
            model_path: Path to .pth model file
            model_name: Model architecture name
            scale: Upscale factor (2 or 4)
            tile_size: Tile size for processing (256 recommended)
            tile_pad: Padding for tiles (10 recommended)
            gpu_id: GPU device ID (None = auto-detect)
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        
        # Validate model file
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Detect device
        self.device, self.use_gpu = self._detect_device(gpu_id)
        
        # Initialize model
        self.upsampler = self._load_model()
        
        print(f"[AI] Device: {self.device}")
        print(f"[AI] Loading model: {self.model_path.name}")
        print(f"[AI] Tile: {tile_size}x{tile_size}, Pad: {tile_pad}")
    
    def _detect_device(self, gpu_id: Optional[int]) -> tuple:
        """Detect CUDA availability and set device."""
        if torch.cuda.is_available():
            device_id = gpu_id if gpu_id is not None else 0
            device = f"cuda:{device_id}"
            gpu_name = torch.cuda.get_device_name(device_id)
            print(f"[AI] GPU acceleration enabled: {gpu_name}")
            return device, True
        else:
            device = "cpu"
            print(f"[AI] CPU fallback (slower)")
            return device, False
    
    def _load_model(self):
        """Load Real-ESRGAN model."""
        # Determine architecture based on model name
        if "anime" in self.model_name.lower():
            # AnimeSR architecture
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,  # AnimeSR uses 6 blocks
                num_grow_ch=32,
                scale=self.scale
            )
        else:
            # Standard RealESRGAN
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=self.scale
            )
        
        # Create upsampler with tile-based processing
        upsampler = RealESRGANer(
            scale=self.scale,
            model_path=str(self.model_path),
            model=model,
            tile=self.tile_size,
            tile_pad=self.tile_pad,
            pre_pad=0,
            half=self.use_gpu,  # FP16 on GPU, FP32 on CPU
            device=self.device
        )
        
        return upsampler
    
    def upscale(
        self,
        image: Image.Image,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Image.Image:
        """
        Upscale image using Real-ESRGAN.
        
        Args:
            image: Input PIL Image (RGB)
            progress_callback: Optional callback(percent) for progress updates
        
        Returns:
            Upscaled PIL Image (RGB)
        """
        # Stage-based progress (Real-ESRGAN doesn't expose real progress)
        if progress_callback:
            progress_callback(20)  # Model loading done, inference starting
        
        # Convert PIL to numpy
        img_np = np.array(image)
        
        if progress_callback:
            progress_callback(40)  # Preprocessing done
        
        # Run Real-ESRGAN inference (tile-based internally)
        try:
            output_np, _ = self.upsampler.enhance(img_np, outscale=self.scale)
        except Exception as e:
            raise RuntimeError(f"Real-ESRGAN inference failed: {e}")
        
        if progress_callback:
            progress_callback(80)  # Inference done
        
        # Convert numpy to PIL
        output_img = Image.fromarray(output_np, mode='RGB')
        
        if progress_callback:
            progress_callback(90)  # Post-processing done
        
        return output_img


# Legacy compatibility wrapper
class AIInferenceEngine:
    """Legacy wrapper for backward compatibility."""
    
    def __init__(
        self,
        model_path: str,
        tile_size: int = 256,
        tile_overlap: int = 10,
        scale_factor: int = 4
    ):
        """Initialize with Real-ESRGAN backend."""
        self.engine = RealESRGANInference(
            model_path=model_path,
            scale=scale_factor,
            tile_size=tile_size,
            tile_pad=tile_overlap
        )
    
    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale image."""
        return self.engine.upscale(image)


def get_device_info() -> dict:
    """Get GPU/CPU device information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu"
    }
    
    if info["cuda_available"]:
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    
    return info

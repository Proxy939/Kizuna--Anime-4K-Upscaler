"""
Real-ESRGAN AI Inference Engine
================================
Handles Real-ESRGAN model loading and inference with GPU/CPU support.

Color Preservation: Processes only Y (lum

inance) through neural network,
upscales Cb/Cr chrominance with bicubic for zero color drift.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    _REALESRGAN_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "Real-ESRGAN dependencies not installed. "
        "Run: pip install realesrgan==0.3.0 basicsr==1.4.2 facexlib gfpgan"
    ) from e


class RealESRGANInference:
    """Real-ESRGAN inference engine with strict color preservation."""
    
    def __init__(
        self,
        model_path: Path,
        tile_size: int = 512,
        tile_pad: int = 10,
        scale_factor: int = 4,
        use_gpu: bool = True
    ):
        """
        Initialize Real-ESRGAN inference engine.
        
        Args:
            model_path: Path to .pth model file
            tile_size: Tile size for processing large images
            tile_pad: Padding for tiles
            scale_factor: Upscaling factor (2 or 4)
            use_gpu: Use GPU if available
        """
        self.model_path = model_path
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.scale = scale_factor
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Validate model file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Set device
        self.device = 'cuda' if self.use_gpu else 'cpu'
        
        # Extract model name
        self.model_name = model_path.stem
        
        # Load model
        self.upsampler = self._load_model()
        
        print(f"[AI] Real-ESRGAN loaded: {self.model_name}")
        print(f"[AI] Device: {self.device.upper()}{' (GPU acceleration enabled)' if self.use_gpu else ''}")
    
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
        Upscale image with STRICT COLOR PRESERVATION.
        
        Strategy:
        1. Convert RGB → YCbCr (separate luminance from color)
        2. Process ONLY Y channel through Real-ESRGAN
        3. Upscale Cb/Cr with bicubic (NO neural processing)
        4. Recombine YCbCr → RGB
        
        Result: ZERO color drift, only spatial resolution improves.
        """
        if progress_callback:
            progress_callback(20)
        
        # ===============================================================
        # STEP 1: Convert to YCbCr
        # ===============================================================
        img_ycbcr = image.convert('YCbCr')
        y_channel, cb_channel, cr_channel = img_ycbcr.split()
        
        if progress_callback:
            progress_callback(30)
        
        # ===============================================================
        # STEP 2: Process Y channel through Real-ESRGAN
        # ===============================================================
        # Convert Y to RGB format for Real-ESRGAN (expects 3 channels)
        y_rgb = Image.merge('RGB', (y_channel, y_channel, y_channel))
        y_array = np.array(y_rgb)
        
        if progress_callback:
            progress_callback(40)
        
        # Real-ESRGAN processing
        output_y, _ = self.upsampler.enhance(y_array, outscale=self.scale)
        
        if progress_callback:
            progress_callback(70)
        
        # Extract first channel (all 3 are identical grayscale)
        output_y_array = np.clip(output_y, 0, 255).astype(np.uint8)
        y_upscaled = Image.fromarray(output_y_array[:, :, 0], mode='L')
        
        # ===============================================================
        # STEP 3: Upscale Cb/Cr with BICUBIC (preserve color perfectly)
        # ===============================================================
        target_size = y_upscaled.size
        cb_upscaled = cb_channel.resize(target_size, Image.BICUBIC)
        cr_upscaled = cr_channel.resize(target_size, Image.BICUBIC)
        
        if progress_callback:
            progress_callback(85)
        
        # ===============================================================
        # STEP 4: Recombine and convert back to RGB
        # ===============================================================
        img_upscaled_ycbcr = Image.merge('YCbCr', (y_upscaled, cb_upscaled, cr_upscaled))
        img_upscaled_rgb = img_upscaled_ycbcr.convert('RGB')
        
        if progress_callback:
            progress_callback(100)
        
        return img_upscaled_rgb


class AIInferenceEngine:
    """High-level AI inference engine for the pipeline."""
    
    def __init__(
        self,
        model_path: Path,
        tile_size: int = 512,
        tile_overlap: int = 10,
        scale_factor: int = 4
    ):
        """Initialize AI inference engine."""
        self.model_path = model_path
        self.tile_size = tile_size
        self.scale_factor = scale_factor
        
        # Initialize Real-ESRGAN engine
        self.engine = RealESRGANInference(
            model_path=model_path,
            tile_size=tile_size,
            tile_pad=tile_overlap,
            scale_factor=scale_factor,
            use_gpu=True
        )
    
    def upscale(self, img: Image.Image) -> Image.Image:
        """Upscale image using Real-ESRGAN."""
        return self.engine.upscale(img)


def get_device_info() -> dict:
    """Get GPU/CPU device information."""
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "cuda_available": False
        }
    
    return {
        "device": "cuda",
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count()
    }

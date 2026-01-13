"""
KizunaSR - AI Inference Module (MVP)
=====================================
Local, offline AI super-resolution using ONNX Runtime.
Supports CUDA, DirectML, and CPU backends with automatic fallback.

This module provides tile-based inference for large images to manage VRAM usage.
"""

import os
import sys
from typing import Tuple, Optional, List
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


class BackendDetector:
    """Detects and ranks available ONNX Runtime execution providers."""
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """
        Query ONNX Runtime for available execution providers.
        
        Returns:
            List of available provider names in priority order.
        """
        available = ort.get_available_providers()
        
        # Define priority order: CUDA > DirectML > CPU
        priority_order = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
        
        # Filter and sort by priority
        providers = [p for p in priority_order if p in available]
        
        if not providers:
            # Fallback to whatever is available
            providers = available
        
        return providers
    
    @staticmethod
    def select_best_provider() -> str:
        """
        Select the best available execution provider.
        
        Returns:
            Name of the selected provider.
        """
        providers = BackendDetector.get_available_providers()
        
        if not providers:
            raise RuntimeError("No ONNX Runtime execution providers available")
        
        selected = providers[0]
        
        # Log selection
        print(f"[AI Backend] Selected: {selected}")
        if len(providers) > 1:
            print(f"[AI Backend] Available fallbacks: {', '.join(providers[1:])}")
        
        return selected


class AIInferenceEngine:
    """
    AI super-resolution inference engine using ONNX Runtime.
    
    Supports tile-based processing for large images and automatic backend selection.
    """
    
    def __init__(
        self,
        model_path: str,
        tile_size: int = 512,
        tile_overlap: int = 64,
        scale_factor: int = 2,
        provider: Optional[str] = None
    ):
        """
        Initialize the AI inference engine.
        
        Args:
            model_path: Path to ONNX model file (.onnx)
            tile_size: Size of tiles for processing (default 512x512)
            tile_overlap: Overlap between tiles in pixels (default 64)
            scale_factor: Upscaling factor of the model (2 or 4)
            provider: Specific execution provider to use (auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.scale_factor = scale_factor
        
        # Validate model file
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Select backend
        if provider is None:
            provider = BackendDetector.select_best_provider()
        
        self.provider = provider
        
        # Load model
        self.session = self._load_model()
        
        print(f"[AI Inference] Model loaded: {self.model_path.name}")
        print(f"[AI Inference] Tile size: {tile_size}x{tile_size}, Overlap: {tile_overlap}px")
        print(f"[AI Inference] Scale factor: {scale_factor}×")
    
    def _load_model(self) -> ort.InferenceSession:
        """
        Load ONNX model with selected execution provider.
        
        Returns:
            ONNX Runtime inference session.
        """
        try:
            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create session with selected provider
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=[self.provider]
            )
            
            # Verify provider was actually used
            actual_provider = session.get_providers()[0]
            if actual_provider != self.provider:
                print(f"[Warning] Requested {self.provider} but using {actual_provider}")
            
            return session
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for AI inference.
        
        Args:
            image: Input image as numpy array (H, W, C) in [0, 255] uint8 RGB
        
        Returns:
            Preprocessed tensor (1, C, H, W) in [0, 1] float32
        """
        # Convert to float32 and normalize to [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW
        img_chw = np.transpose(img_float, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch
    
    def postprocess(self, tensor: np.ndarray) -> np.ndarray:
        """
        Postprocess AI output tensor to image.
        
        Args:
            tensor: Output tensor (1, C, H, W) in [0, 1] float32
        
        Returns:
            Image as numpy array (H, W, C) in [0, 255] uint8 RGB
        """
        # Remove batch dimension
        img_chw = tensor[0]
        
        # Convert from CHW to HWC
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        
        # Denormalize from [0, 1] to [0, 255]
        img_uint8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)
        
        return img_uint8
    
    def infer_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Run inference on a single tile.
        
        Args:
            tile: Preprocessed tile tensor (1, C, H, W)
        
        Returns:
            Upscaled tile tensor (1, C, H*scale, W*scale)
        """
        # Get input name from model
        input_name = self.session.get_inputs()[0].name
        
        # Run inference
        outputs = self.session.run(None, {input_name: tile})
        
        return outputs[0]
    
    def extract_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract overlapping tiles from image.
        
        Args:
            image: Input image (H, W, C)
        
        Returns:
            List of (tile, (x, y, w, h)) where tile is preprocessed and (x,y,w,h) is position
        """
        h, w, _ = image.shape
        tiles = []
        
        stride = self.tile_size - self.tile_overlap
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Calculate tile boundaries
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)
                
                # Extract tile
                tile_img = image[y:y_end, x:x_end, :]
                
                # Pad if necessary (for edge tiles)
                if tile_img.shape[0] < self.tile_size or tile_img.shape[1] < self.tile_size:
                    pad_h = self.tile_size - tile_img.shape[0]
                    pad_w = self.tile_size - tile_img.shape[1]
                    tile_img = np.pad(
                        tile_img,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode='reflect'
                    )
                
                # Preprocess tile
                tile_tensor = self.preprocess(tile_img)
                
                # Store tile and its position
                tiles.append((tile_tensor, (x, y, x_end - x, y_end - y)))
        
        return tiles
    
    def blend_tiles(
        self,
        tiles: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
        output_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Blend overlapping tiles into final output image.
        
        Args:
            tiles: List of (upscaled_tile_tensor, (x, y, w, h))
            output_shape: Shape of final output (H, W, C)
        
        Returns:
            Blended output image (H, W, C) uint8
        """
        h, w, c = output_shape
        
        # Create output canvas and weight map
        output = np.zeros((h, w, c), dtype=np.float32)
        weights = np.zeros((h, w, 1), dtype=np.float32)
        
        # Process each tile
        for tile_tensor, (x, y, tw, th) in tiles:
            # Postprocess tile
            tile_img = self.postprocess(tile_tensor)
            
            # Calculate output position (scaled)
            out_x = x * self.scale_factor
            out_y = y * self.scale_factor
            out_w = tw * self.scale_factor
            out_h = th * self.scale_factor
            
            # Crop tile to actual size (remove padding if any)
            tile_crop = tile_img[:out_h, :out_w, :]
            
            # Create feathered weight mask for blending
            weight_mask = self._create_blend_mask(out_h, out_w)
            
            # Accumulate tile into output
            output[out_y:out_y+out_h, out_x:out_x+out_w, :] += tile_crop * weight_mask
            weights[out_y:out_y+out_h, out_x:out_x+out_w, :] += weight_mask
        
        # Normalize by weights
        output = output / np.maximum(weights, 1e-8)
        
        return output.astype(np.uint8)
    
    def _create_blend_mask(self, h: int, w: int) -> np.ndarray:
        """
        Create feathered blend mask for tile edges.
        
        Args:
            h: Tile height
            w: Tile width
        
        Returns:
            Weight mask (H, W, C) with smooth falloff at edges
        """
        # Create 1D ramps for each dimension
        ramp_h = np.ones(h, dtype=np.float32)
        ramp_w = np.ones(w, dtype=np.float32)
        
        # Feather width (half of overlap)
        feather = self.tile_overlap // 2
        
        # Apply feathering at edges
        if feather > 0:
            for i in range(min(feather, h)):
                ramp_h[i] = i / feather
                if i < h:
                    ramp_h[-(i+1)] = i / feather
            
            for i in range(min(feather, w)):
                ramp_w[i] = i / feather
                if i < w:
                    ramp_w[-(i+1)] = i / feather
        
        # Create 2D mask by outer product
        mask_2d = np.outer(ramp_h, ramp_w)
        
        # Expand to 3 channels
        mask_3d = np.stack([mask_2d] * 3, axis=-1)
        
        return mask_3d
    
    def upscale(self, image: Image.Image) -> Image.Image:
        """
        Upscale an image using tile-based AI inference.
        
        Args:
            image: PIL Image (RGB)
        
        Returns:
            Upscaled PIL Image (RGB)
        """
        # Convert to numpy
        img_np = np.array(image)
        
        if img_np.shape[2] != 3:
            raise ValueError("Input image must be RGB (3 channels)")
        
        h, w, c = img_np.shape
        
        print(f"[AI Inference] Processing image: {w}×{h}")
        
        # Extract tiles
        tiles = self.extract_tiles(img_np)
        print(f"[AI Inference] Extracted {len(tiles)} tiles")
        
        # Process each tile
        upscaled_tiles = []
        for i, (tile_tensor, pos) in enumerate(tiles):
            if (i + 1) % 10 == 0 or (i + 1) == len(tiles):
                print(f"[AI Inference] Processing tile {i+1}/{len(tiles)}")
            
            # Run inference
            upscaled_tensor = self.infer_tile(tile_tensor)
            upscaled_tiles.append((upscaled_tensor, pos))
        
        # Blend tiles
        output_shape = (h * self.scale_factor, w * self.scale_factor, c)
        output_np = self.blend_tiles(upscaled_tiles, output_shape)
        
        # Convert back to PIL Image
        output_img = Image.fromarray(output_np, mode='RGB')
        
        print(f"[AI Inference] Output: {output_img.width}×{output_img.height}")
        
        return output_img


def main():
    """Example usage of AI Inference Engine."""
    
    # Example configuration
    model_path = "models/anime_sr_2x.onnx"  # User must provide this
    input_image_path = "input.png"
    output_image_path = "output.png"
    
    try:
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"[Error] Model not found: {model_path}")
            print("[Info] Please download an anime SR ONNX model and place it in 'models/' directory")
            print("[Info] Recommended: RealESRGAN-anime or AnimeSR models converted to ONNX")
            return
        
        # Check if input exists
        if not os.path.exists(input_image_path):
            print(f"[Error] Input image not found: {input_image_path}")
            return
        
        # Initialize engine
        engine = AIInferenceEngine(
            model_path=model_path,
            tile_size=512,
            tile_overlap=64,
            scale_factor=2
        )
        
        # Load input image
        input_img = Image.open(input_image_path).convert('RGB')
        
        # Upscale
        output_img = engine.upscale(input_img)
        
        # Save result
        output_img.save(output_image_path)
        print(f"[Success] Saved upscaled image to: {output_image_path}")
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")
    except RuntimeError as e:
        print(f"[Error] {e}")
    except Exception as e:
        print(f"[Error] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

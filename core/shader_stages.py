"""
KizunaSR - CPU Reference Shader Stages
=======================================
Pure Python/NumPy implementations of the shader pipeline stages.
These are reference implementations to validate architecture and data flow.

These will later be replaced by actual GPU shaders for real-time performance.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """
    Convert sRGB to linear RGB color space.
    
    Args:
        image: RGB image (H, W, 3) in [0, 255] uint8
    
    Returns:
        Linear RGB image (H, W, 3) in [0, 1] float32
    """
    img_float = image.astype(np.float32) / 255.0
    
    # Apply sRGB gamma correction
    linear = np.where(
        img_float <= 0.04045,
        img_float / 12.92,
        np.power((img_float + 0.055) / 1.055, 2.4)
    )
    
    return linear.astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB to sRGB color space.
    
    Args:
        image: Linear RGB (H, W, 3) in [0, 1] float32
    
    Returns:
        sRGB image (H, W, 3) in [0, 255] uint8
    """
    # Apply inverse sRGB gamma
    srgb = np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055
    )
    
    # Clamp and convert to uint8
    srgb = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
    
    return srgb


# ==============================================================================
# Stage 1: Normalize
# ==============================================================================

def stage_normalize(image: np.ndarray) -> np.ndarray:
    """
    Stage 1: Normalize
    
    Convert input to linear RGB and apply light denoising.
    
    Args:
        image: Input RGB image (H, W, 3) uint8 [0, 255]
    
    Returns:
        Normalized linear RGB (H, W, 3) float32 [0, 1]
    """
    # Convert to linear RGB
    linear = srgb_to_linear(image)
    
    # Apply subtle Gaussian denoising (sigma=0.5 for compression artifacts)
    denoised = gaussian_filter(linear, sigma=0.5)
    
    # Clamp to valid range
    result = np.clip(denoised, 0.0, 1.0)
    
    return result


# ==============================================================================
# Stage 2: Structural Reconstruction
# ==============================================================================

def detect_edges_sobel(image: np.ndarray) -> np.ndarray:
    """
    Detect edges using Sobel operator on luminance.
    
    Args:
        image: RGB image (H, W, 3) float32
    
    Returns:
        Edge strength map (H, W) float32 [0, 1]
    """
    # Convert to luminance
    luma = np.dot(image, [0.299, 0.587, 0.114])
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Convolve with Sobel kernels
    from scipy.signal import convolve2d
    grad_x = convolve2d(luma, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(luma, sobel_y, mode='same', boundary='symm')
    
    # Gradient magnitude
    edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to [0, 1]
    edge_strength = np.clip(edge_strength / 0.5, 0.0, 1.0)
    
    return edge_strength


def stage_structural_reconstruction(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 2: Structural Reconstruction
    
    Detect edges and apply edge-aware smoothing.
    
    Args:
        image: Normalized linear RGB (H, W, 3) float32
    
    Returns:
        Tuple of (processed_image, edge_map)
        - processed_image: RGB (H, W, 3) float32
        - edge_map: Edge strength (H, W) float32 [0, 1]
    """
    # Detect edges
    edge_map = detect_edges_sobel(image)
    
    # Edge-aware smoothing: smooth weak edges, preserve strong edges
    smoothing_strength = 0.3
    smoothed = gaussian_filter(image, sigma=0.8)
    
    # Blend based on edge strength (strong edges = less smoothing)
    edge_factor = 1.0 - edge_map[..., np.newaxis]  # Invert for preservation
    processed = image * edge_map[..., np.newaxis] + smoothed * edge_factor * smoothing_strength
    processed = processed + image * (1.0 - smoothing_strength)
    
    # Normalize
    processed = (processed - processed.min()) / (processed.max() - processed.min() + 1e-8)
    
    return processed, edge_map


# ==============================================================================
# Stage 3: Real-Time Upscale (Analytic, not AI)
# ==============================================================================

def bicubic_upscale(image: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Bicubic interpolation upscaling.
    
    Args:
        image: Input image (H, W, C) float32
        scale: Upscaling factor
    
    Returns:
        Upscaled image (H*scale, W*scale, C) float32
    """
    from PIL import Image
    
    h, w, c = image.shape
    
    # Convert to PIL for bicubic resize
    img_uint8 = (image * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    
    # Bicubic resize
    upscaled_pil = pil_img.resize((w * scale, h * scale), Image.BICUBIC)
    
    # Convert back to float32
    upscaled = np.array(upscaled_pil).astype(np.float32) / 255.0
    
    return upscaled


def stage_realtime_upscale(
    image: np.ndarray,
    edge_map: np.ndarray,
    scale: int = 2,
    edge_aware: bool = True
) -> np.ndarray:
    """
    Stage 3: Real-Time Upscale
    
    Analytic upscaling (bicubic) with optional edge-awareness.
    In GPU version, this would be edge-directed interpolation.
    
    Args:
        image: Processed image (H, W, 3) float32
        edge_map: Edge strength map (H, W) float32
        scale: Upscaling factor (default 2)
        edge_aware: Whether to use edge information (default True)
    
    Returns:
        Upscaled image (H*scale, W*scale, 3) float32
    """
    # Standard bicubic upscaling
    upscaled = bicubic_upscale(image, scale=scale)
    
    # In a full GPU implementation, we would use edge_map for directional sampling.
    # For this CPU reference, bicubic is sufficient.
    
    return upscaled


# ==============================================================================
# Stage 4: Perceptual Enhancement
# ==============================================================================

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV color space."""
    from matplotlib.colors import rgb_to_hsv as mpl_rgb_to_hsv
    return mpl_rgb_to_hsv(rgb)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV to RGB color space."""
    from matplotlib.colors import hsv_to_rgb as mpl_hsv_to_rgb
    return mpl_hsv_to_rgb(hsv)


def stage_perceptual_enhancement(
    image: np.ndarray,
    contrast_boost: float = 1.1,
    saturation_boost: float = 1.05,
    sharpening: float = 0.3,
    line_darkening: float = 0.15
) -> np.ndarray:
    """
    Stage 4: Perceptual Enhancement
    
    Apply contrast, saturation, sharpening, and line art darkening.
    
    Args:
        image: Upscaled image (H, W, 3) float32
        contrast_boost: Contrast multiplier (1.0 = no change)
        saturation_boost: Saturation multiplier (1.0 = no change)
        sharpening: Sharpening strength (0.0 = none)
        line_darkening: Line darkening strength (0.0 = none)
    
    Returns:
        Enhanced image (H, W, 3) float32
    """
    result = image.copy()
    
    # 1. Sharpening (unsharp mask)
    if sharpening > 0.01:
        blurred = gaussian_filter(result, sigma=1.0)
        detail = result - blurred
        result = result + detail * sharpening
        result = np.clip(result, 0.0, 1.0)
    
    # 2. Line darkening (darken dark pixels)
    if line_darkening > 0.01:
        luma = np.dot(result, [0.299, 0.587, 0.114])
        dark_mask = (luma < 0.3).astype(np.float32)
        darken_factor = 1.0 - line_darkening * dark_mask[..., np.newaxis]
        result = result * darken_factor
        result = np.clip(result, 0.0, 1.0)
    
    # 3. Contrast adjustment
    if abs(contrast_boost - 1.0) > 0.01:
        result = (result - 0.5) * contrast_boost + 0.5
        result = np.clip(result, 0.0, 1.0)
    
    # 4. Saturation boost
    if abs(saturation_boost - 1.0) > 0.01:
        hsv = rgb_to_hsv(result)
        hsv[..., 1] *= saturation_boost  # Boost saturation
        hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 1.0)
        result = hsv_to_rgb(hsv)
        result = np.clip(result, 0.0, 1.0)
    
    return result


# ==============================================================================
# Stage 5: Temporal Stabilization
# ==============================================================================

def stage_temporal_stabilization(
    current_frame: np.ndarray,
    previous_frame: np.ndarray,
    temporal_weight: float = 0.15,
    motion_threshold: float = 0.1
) -> np.ndarray:
    """
    Stage 5: Temporal Stabilization
    
    Blend current frame with previous frame to reduce flicker.
    
    Args:
        current_frame: Current processed frame (H, W, 3) float32
        previous_frame: Previous output frame (H, W, 3) float32 or None
        temporal_weight: Blend weight for history (0.0 = no temporal)
        motion_threshold: Pixel difference threshold for motion detection
    
    Returns:
        Stabilized frame (H, W, 3) float32
    """
    if previous_frame is None or temporal_weight < 0.01:
        return current_frame
    
    # Detect motion via pixel difference
    diff = np.abs(current_frame - previous_frame)
    motion = np.max(diff, axis=-1)  # Max across RGB channels
    
    # Adaptive blending: static regions blend, moving regions don't
    # Use smoothstep-like function
    blend_factor = np.clip((motion - motion_threshold * 0.5) / (motion_threshold), 0.0, 1.0)
    history_weight = temporal_weight * (1.0 - blend_factor)
    
    # Blend
    result = current_frame * (1.0 - history_weight[..., np.newaxis]) + \
             previous_frame * history_weight[..., np.newaxis]
    
    return result

# KizunaSR AI Inference Module - MVP

## Overview
This module implements local, offline AI super-resolution using ONNX Runtime with automatic GPU/CPU backend selection and tile-based processing for large images.

## Features
- **Local-first**: All inference runs on user's own hardware, no cloud
- **Multi-backend**: CUDA (NVIDIA) → DirectML (Windows fallback) → CPU (last resort)
- **Tile-based**: Handles large images by splitting into tiles with overlap
- **Seam-free blending**: Feathered blending prevents visible tile boundaries
- **Error handling**: Graceful fallback when hardware is insufficient

## Installation

### 1. Install Dependencies

**For NVIDIA GPU (CUDA)**:
```bash
pip install -r requirements.txt
pip install onnxruntime-gpu
```

**For AMD/Intel GPU or CPU-only**:
```bash
pip install onnxruntime numpy Pillow
```

**For DirectML (Windows, any GPU)**:
```bash
pip install onnxruntime-directml numpy Pillow
```

### 2. Download AI Model

KizunaSR requires a pre-trained anime super-resolution model in ONNX format.

**Recommended Models**:
- **RealESRGAN-anime**: Excellent for anime, available in 2× and 4× variants
- **AnimeSR**: Lightweight, good quality-to-speed ratio
- **SwinIR-anime**: Higher quality, slower inference

**Where to Get Models**:
1. Search for "RealESRGAN anime ONNX" or "AnimeSR ONNX" on GitHub/Hugging Face
2. Or convert PyTorch models to ONNX yourself using `torch.onnx.export()`

**Model Placement**:
```
Kizuna (絆)/
├── models/
│   └── anime_sr_2x.onnx   <-- Place your model here
└── runtime/
    └── ai/
        └── ai_inference.py
```

## Usage

### Basic Usage

```python
from runtime.ai.ai_inference import AIInferenceEngine
from PIL import Image

# Initialize engine
engine = AIInferenceEngine(
    model_path="models/anime_sr_2x.onnx",
    tile_size=512,        # 512×512 tiles (adjust based on VRAM)
    tile_overlap=64,      # 64px overlap for seamless blending
    scale_factor=2        # Model's upscale factor (2× or 4×)
)

# Load and upscale image
input_img = Image.open("input.png").convert('RGB')
output_img = engine.upscale(input_img)
output_img.save("output.png")
```

### Command-Line Usage

```bash
# Edit model_path and input_image_path in ai_inference.py, then:
python runtime/ai/ai_inference.py
```

### Advanced Configuration

**Adjust Tile Size (VRAM Management)**:
```python
# High VRAM (8+ GB): Larger tiles, fewer seams
engine = AIInferenceEngine(model_path, tile_size=768, tile_overlap=96)

# Low VRAM (4-6 GB): Standard tiles
engine = AIInferenceEngine(model_path, tile_size=512, tile_overlap=64)

# Very Low VRAM (2-4 GB): Smaller tiles
engine = AIInferenceEngine(model_path, tile_size=256, tile_overlap=32)
```

**Force Specific Backend**:
```python
# Force CPU (for testing or if GPU fails)
engine = AIInferenceEngine(model_path, provider='CPUExecutionProvider')

# Force CUDA
engine = AIInferenceEngine(model_path, provider='CUDAExecutionProvider')

# Force DirectML (Windows)
engine = AIInferenceEngine(model_path, provider='DmlExecutionProvider')
```

## Architecture

### Backend Detection
The `BackendDetector` class probes ONNX Runtime for available execution providers and selects the best one:

**Priority Order**:
1. **CUDAExecutionProvider** (NVIDIA GPUs with CUDA)
2. **DmlExecutionProvider** (DirectML for Windows, any GPU)
3. **CPUExecutionProvider** (always available)

### Tile-Based Inference Pipeline

```
Input Image → Extract Tiles (with overlap)
             ↓
         Process Each Tile (AI inference)
             ↓
         Blend Tiles (feathered weights)
             ↓
         Output Image
```

**Why Tile-Based?**
- A single 1080p→4K inference can require 4+ GB VRAM
- Tiling keeps VRAM usage predictable (~500 MB per tile)
- Enables arbitrary resolution processing

### Tile Blending Strategy

**Overlap**: Tiles overlap by `tile_overlap` pixels (default 64px)

**Feathered Blending**:
- Each tile has a weight mask that fades from 1.0 at center to 0.0 at edges
- Overlapping regions are averaged using these weights
- Result: Seamless transitions, no visible tile boundaries

**Blend Mask Formula**:
```
For each edge pixel i in overlap region (width = feather):
    weight[i] = i / feather
```

## Implementation Details

### Preprocessing
```python
# Input: PIL Image RGB (H, W, 3) uint8 [0, 255]
# ↓
# Normalize to float32 [0, 1]
# ↓
# Transpose HWC → CHW
# ↓
# Add batch dimension → (1, C, H, W)
# Output: Ready for ONNX model
```

### Inference
```python
# Get input name from model metadata
# Run session.run(None, {input_name: tensor})
# Output: (1, C, H*scale, W*scale) float32 [0, 1]
```

### Postprocessing
```python
# Input: ONNX output (1, C, H, W)
# ↓
# Remove batch dimension
# ↓
# Transpose CHW → HWC
# ↓
# Denormalize [0, 1] → [0, 255] uint8
# Output: PIL Image RGB
```

## Validation and Testing

### Visual Validation

**Expected Results**:
- Sharp line art (model should enhance edges)
- No visible tile seams or boundaries
- Smooth gradients without banding
- Enhanced details without over-sharpening artifacts

**Common Artifacts**:
- **Tile seams**: Visible grid pattern → Increase overlap or check blending
- **Over-sharpening**: Ringing/halos around edges → Model issue, try different model
- **Blurry output**: Model not working → Check preprocessing normalization
- **Color shifts**: Incorrect color space → Ensure RGB input/output

### Tile Blending Verification

**Test with problematic content**:
1. Solid color image → Should remain uniform (no seams)
2. Smooth gradient → Should be continuous (no tile grid visible)
3. Sharp edges crossing tiles → Edges should remain crisp and aligned

**Debug tile positions**:
```python
# After extract_tiles(), print tile positions
for i, (_, (x, y, w, h)) in enumerate(tiles):
    print(f"Tile {i}: x={x}, y={y}, size={w}×{h}")
```

### Performance Benchmarks

**Estimated Inference Time** (RTX 3060, 512×512 tiles):
- 1080p → 4K (9 tiles): ~5-10 seconds
- 720p → 1440p (4 tiles): ~2-4 seconds

**VRAM Usage**:
- Per-tile: ~500 MB
- Total: Model size (20-100 MB) + Tile buffer (~500 MB) = ~1 GB

### Error Handling Test Cases

**Test 1: Missing Model File**
```bash
# Run with non-existent model path
# Expected: FileNotFoundError with clear message
```

**Test 2: Unsupported Backend**
```python
# Force an unsupported provider
engine = AIInferenceEngine(model_path, provider='NonExistentProvider')
# Expected: RuntimeError with fallback message
```

**Test 3: Invalid Input**
```python
# Provide grayscale image
img_gray = Image.open("input.png").convert('L')
output = engine.upscale(img_gray)
# Expected: ValueError "Input image must be RGB"
```

## Model Format Requirements

### ONNX Model Specifications

**Input**:
- Name: Typically `input` or `lr` (low-resolution)
- Shape: `(1, 3, H, W)` or dynamic `(1, 3, ?, ?)`
- Type: Float32
- Range: `[0.0, 1.0]` (normalized RGB)

**Output**:
- Name: Typically `output` or `hr` (high-resolution)
- Shape: `(1, 3, H*scale, W*scale)`
- Type: Float32
- Range: `[0.0, 1.0]`

**Supported Scale Factors**: 2×, 4× (set `scale_factor` parameter accordingly)

### Converting PyTorch Models to ONNX

```python
import torch
import torch.onnx

# Load your PyTorch model
model = YourSRModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 256, 256)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {2: 'height', 3: 'width'},
                  'output': {2: 'height', 3: 'width'}},
    opset_version=14
)
```

## Limitations and Future Work

### Current Limitations
- Single-image only (no video support yet)
- No temporal AI (each frame processed independently)
- Fixed tile size (no adaptive sizing based on content)
- No multi-GPU support

### Future Improvements
- Adaptive tile sizing based on VRAM availability
- Batch processing for multiple images
- Video integration (frame-by-frame with progress)
- Model zoo management (download and cache models)
- GPU memory profiling and auto-tuning

## Troubleshooting

### "No execution providers available"
- Install onnxruntime-gpu (for CUDA) or onnxruntime-directml (for DirectML)
- Check GPU drivers are up to date

### "Out of memory" during inference
- Reduce tile_size: 512 → 256
- Close other GPU applications
- Fall back to CPU: `provider='CPUExecutionProvider'`

### Slow CPU inference
- CPU inference is 10-50× slower than GPU
- Consider using smaller model or lower resolution
- Or wait for GPU, CPU is only a fallback

### Tile seams visible in output
- Increase tile_overlap: 64 → 128
- Check blend mask implementation
- Verify tile positions don't have gaps

---

## Conclusion

This MVP implementation demonstrates that KizunaSR can perform robust, local AI super-resolution on anime content using the user's own hardware. The tile-based architecture ensures scalability to arbitrary resolutions, and the multi-backend support provides broad hardware compatibility.

The implementation prioritizes correctness and safety over performance, serving as a foundation for future optimization and integration with the shader pipeline and video processing systems.

---
*AI Inference Module Version: 1.0 MVP*  
*Status: Complete - Ready for Testing and Integration*

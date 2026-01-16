"""
Central registry for AI models.
"""

MODELS = {
    "realesrgan-x4plus": {
        "key": "realesrgan-x4plus",
        "label": "Real-ESRGAN x4+ Anime (Default ⭐)",
        "filename": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "default": True
    }
}

def get_model(key: str):
    """Get model config by key."""
    return MODELS.get(key)

def get_default_model():
    """Get the default model config."""
    for model in MODELS.values():
        if model.get("default"):
            return model
    return None

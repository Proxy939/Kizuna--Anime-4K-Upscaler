"""
Model Registry - Single Source of Truth for AI Models
"""

MODELS = {
    "realesrgan-anime-v3": {
        "key": "realesrgan-anime-v3",
        "label": "Real-ESRGAN Anime (Default ⭐)",
        "filename": "realesrgan-animevideov3.pth",
        "scale": 4,
        "default": True
    },
    "realesrgan-anime-x4plus": {
        "key": "realesrgan-anime-x4plus",
        "label": "Real-ESRGAN Anime x4+",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "default": False
    },
    "realesrgan-general-x4plus": {
        "key": "realesrgan-general-x4plus",
        "label": "Real-ESRGAN General x4+",
        "filename": "RealESRGAN_x4plus.pth",
        "scale": 4,
        "default": False
    }
}


def get_default_model():
    """Get the default model configuration."""
    for m in MODELS.values():
        if m.get("default"):
            return m
    return None


def get_model(key: str):
    """Get model configuration by key."""
    return MODELS.get(key)

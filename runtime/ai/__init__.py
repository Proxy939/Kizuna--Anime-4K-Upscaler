"""
KizunaSR - AI Inference Module Package
"""

from .model_registry import MODELS, get_default_model, get_model
from .ai_inference import AIInferenceEngine, get_device_info

__all__ = ['AIInferenceEngine', 'MODELS', 'get_default_model', 'get_model', 'get_device_info']
__version__ = '1.0.0-mvp'

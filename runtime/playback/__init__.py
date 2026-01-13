"""KizunaSR - Real-Time Playback Engine"""

from .window import PlaybackWindow
from .texture_manager import OpenGLTextureUploader
from .shader_pipeline import ShaderPipelineExecutor

__all__ = ['PlaybackWindow', 'OpenGLTextureUploader', 'ShaderPipelineExecutor']
__version__ = '1.0.0-mvp'

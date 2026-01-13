"""KizunaSR - Real-Time Playback Engine"""

from .window import PlaybackWindow
from .texture_manager import OpenGLTextureUploader
from .shader_pipeline import ShaderPipelineExecutor
from .scheduler import FrameScheduler, Clock, SystemClock, AudioClock

__all__ = [
    'PlaybackWindow',
    'OpenGLTextureUploader',
    'ShaderPipelineExecutor',
    'FrameScheduler',
    'Clock',
    'SystemClock',
    'AudioClock'
]
__version__ = '1.0.0-mvp'

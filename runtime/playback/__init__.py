"""KizunaSR - Real-Time Playback Engine"""

from .window import PlaybackWindow
from .texture_manager import OpenGLTextureUploader
from .shader_pipeline import ShaderPipelineExecutor
from .scheduler import FrameScheduler, Clock, SystemClock, AudioClock
from .audio_player import AudioPlayer

__all__ = [
    'PlaybackWindow',
    'OpenGLTextureUploader',
    'ShaderPipelineExecutor',
    'FrameScheduler',
    'Clock',
    'SystemClock',
    'AudioClock',
    'AudioPlayer'
]
__version__ = '1.0.0-mvp'

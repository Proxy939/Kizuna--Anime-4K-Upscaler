"""
KizunaSR - Real-Time Player
============================
Main playback engine integrating all components.

Orchestrates:
- Video decoding
- Audio playback & master clock
- Frame scheduling
- GPU shader pipeline
- Window rendering

This is the final assembly of the real-time playback system.
"""

import threading
from pathlib import Path
from typing import Optional

from runtime.video import VideoDecoder
from runtime.playback import (
    PlaybackWindow,
    OpenGLTextureUploader,
    ShaderPipelineExecutor,
    FrameScheduler,
    AudioPlayer
)


class Player:
    """
    Real-time video player with GPU shader pipeline.
    
    Integrates all KizunaSR components for smooth A/V playback.
    """
    
    def __init__(
        self,
        video_path: str,
        scale_factor: int = 2,
        use_vsync: bool = True,
        shader_dir: Optional[str] = None
    ):
        """
        Initialize player.
        
        Args:
            video_path: Path to video file
            scale_factor: Upscaling factor (2 or 4)
            use_vsync: Enable V-sync for smooth playback
            shader_dir: Directory containing GLSL shaders (auto-detected if None)
        """
        self.video_path = Path(video_path)
        self.scale_factor = scale_factor
        self.use_vsync = use_vsync
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Auto-detect shader directory
        if shader_dir is None:
            shader_dir = Path(__file__).parent.parent / "shaders" / "gpu"
        self.shader_dir = Path(shader_dir)
        
        # Components (initialized in play())
        self.window: Optional[PlaybackWindow] = None
        self.uploader: Optional[OpenGLTextureUploader] = None
        self.pipeline: Optional[ShaderPipelineExecutor] = None
        self.audio: Optional[AudioPlayer] = None
        self.scheduler: Optional[FrameScheduler] = None
        self.decoder: Optional[VideoDecoder] = None
        
        # Threading
        self.decode_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # State
        self.running = False
        
        print("=" * 70)
        print("KizunaSR Real-Time Player")
        print("=" * 70)
        print(f"Video:  {self.video_path.name}")
        print(f"Scale:  {scale_factor}×")
        print(f"V-sync: {'ON' if use_vsync else 'OFF'}")
        print("=" * 70)
    
    def play(self):
        """Start playback (blocking until window closes)."""
        try:
            self._initialize()
            self._start_threads()
            self._run_render_loop()
        finally:
            self._cleanup()
    
    def stop(self):
        """Stop playback."""
        self.running = False
        self.stop_event.set()
    
    def _initialize(self):
        """Initialize all components in correct order."""
        print("\n[Player] Initializing components...")
        
        # 1. Create window (provides OpenGL context)
        print("[Player] Creating window...")
        self.window = PlaybackWindow(
            width=1280,
            height=720,
            title=f"KizunaSR - {self.video_path.name}",
            vsync=self.use_vsync
        )
        
        # 2. Initialize texture uploader (requires OpenGL context)
        print("[Player] Initializing texture uploader...")
        self.uploader = OpenGLTextureUploader(enable_texture_reuse=True)
        
        # 3. Initialize shader pipeline
        print("[Player] Loading shader pipeline...")
        self.pipeline = ShaderPipelineExecutor(
            shader_dir=str(self.shader_dir),
            scale_factor=self.scale_factor
        )
        
        # 4. Open audio player (may fail gracefully)
        print("[Player] Opening audio stream...")
        self.audio = AudioPlayer()
        has_audio = self.audio.open(str(self.video_path))
        
        if has_audio:
            print("[Player] Audio stream opened successfully")
        else:
            print("[Player] No audio stream (video-only playback)")
        
        # 5. Get audio clock (master clock)
        audio_clock = self.audio.get_clock()
        
        # 6. Create frame scheduler (uses audio clock)
        print("[Player] Initializing frame scheduler...")
        self.scheduler = FrameScheduler(
            master_clock=audio_clock,
            max_queue_size=16,
            lateness_threshold=0.05
        )
        
        # 7. Create video decoder
        print("[Player] Opening video decoder...")
        self.decoder = VideoDecoder(str(self.video_path))
        
        # 8. Resize pipeline for input resolution
        self.pipeline.resize(self.decoder.width, self.decoder.height)
        
        print(f"\n[Player] Initialization complete")
        print(f"[Player] Input:  {self.decoder.width}×{self.decoder.height}")
        print(f"[Player] Output: {self.decoder.width * self.scale_factor}×{self.decoder.height * self.scale_factor}")
    
    def _start_threads(self):
        """Start decoder and audio threads."""
        print("\n[Player] Starting threads...")
        
        self.running = True
        self.stop_event.clear()
        
        # Start audio playback
        self.audio.start()
        
        # Start decoder thread
        self.decode_thread = threading.Thread(
            target=self._decode_loop,
            name="DecoderThread",
            daemon=True
        )
        self.decode_thread.start()
        
        print("[Player] Threads started")
    
    def _decode_loop(self):
        """Decoder thread: decode frames and push to scheduler."""
        print("[Decoder] Thread started")
        
        try:
            for frame in self.decoder:
                if self.stop_event.is_set():
                    break
                
                # Push frame to scheduler (blocks if queue full)
                self.scheduler.push_frame(frame, block=True, timeout=1.0)
                
                # Check stop condition
                if not self.running:
                    break
        
        except Exception as e:
            print(f"[Decoder] Error: {e}")
        
        finally:
            print("[Decoder] Thread finished")
    
    def _run_render_loop(self):
        """Main render loop (runs on main thread)."""
        print("\n[Player] Starting playback...")
        print("[Player] Press ESC to stop\n")
        
        frame_count = 0
        
        def render_frame():
            """Per-frame rendering callback."""
            nonlocal frame_count
            
            # Get next frame from scheduler
            video_frame = self.scheduler.get_next_frame()
            
            if video_frame is not None:
                # Upload frame to GPU
                input_texture = self.uploader.upload_frame(video_frame)
                
                # Execute shader pipeline
                output_texture = self.pipeline.execute(
                    input_texture,
                    frame_index=video_frame.frame_index
                )
                
                # Render output texture to screen
                # (For now, we'll just let the pipeline render to FBO)
                # A final blit step would display the output texture
                # For MVP, the pipeline's last stage output is already rendered
                
                # Simple display: bind and render the output texture as fullscreen quad
                self._display_texture(output_texture)
                
                # Release input texture back to pool
                self.uploader.release_texture(input_texture)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    stats = self.scheduler.get_stats()
                    print(f"[Player] Frame {frame_count} | "
                          f"Presented: {stats.presented_frames} | "
                          f"Dropped: {stats.dropped_frames}")
        
        # Run window's main loop with our render callback
        self.window.run(frame_callback=render_frame)
        
        # Print final stats
        stats = self.scheduler.get_stats()
        print(f"\n[Player] Playback finished")
        print(f"[Player] Total frames: {stats.total_frames}")
        print(f"[Player] Presented: {stats.presented_frames}")
        print(f"[Player] Dropped: {stats.dropped_frames}")
    
    def _display_texture(self, texture_id: int):
        """
        Display a texture to the screen as fullscreen quad.
        
        Args:
            texture_id: OpenGL texture ID to display
        """
        from OpenGL.GL import *
        
        # For MVP: We'd need a simple blit shader to display the final texture
        # Since our pipeline already renders to FBOs, we can blit the final FBO to screen
        
        # Simplified approach: The pipeline's last stage output is in an FBO
        # We need to blit that FBO to the default framebuffer (screen)
        
        # Get output texture size
        width, height = self.uploader.get_texture_size(texture_id)
        
        # Blit from pipeline's final FBO to default framebuffer
        # (This requires knowing which FBO contains our output texture)
        
        # For now, we'll use a simple approach: bind texture and render fullscreen quad
        # We need a simple passthrough shader for this
        
        # TODO: Implement final display pass
        # For MVP, just clear to show something is happening
        glClear(GL_COLOR_BUFFER_BIT)
    
    def _cleanup(self):
        """Clean up all components."""
        print("\n[Player] Cleaning up...")
        
        # Stop threads
        self.running = False
        self.stop_event.set()
        
        if self.decode_thread and self.decode_thread.is_alive():
            self.decode_thread.join(timeout=2.0)
        
        # Stop audio
        if self.audio:
            self.audio.stop()
        
        # Cleanup GPU resources
        if self.uploader:
            self.uploader.cleanup()
        
        if self.pipeline:
            self.pipeline.cleanup()
        
        # Close decoder
        if self.decoder:
            self.decoder.close()
        
        # Close window (if using context manager, already done)
        
        print("[Player] Cleanup complete")


def main():
    """Example usage."""
    import sys
    
    # Check for input file
    video_path = "input.mp4"
    
    if not Path(video_path).exists():
        print(f"[Error] Video file not found: {video_path}")
        print("[Info] Place a video file as 'input.mp4' to test playback")
        return 1
    
    try:
        # Create and run player
        player = Player(
            video_path=video_path,
            scale_factor=2,
            use_vsync=True
        )
        
        player.play()
        
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user")
    except Exception as e:
        print(f"\n[Error] Playback failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

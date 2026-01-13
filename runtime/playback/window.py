"""
KizunaSR - Playback Window & OpenGL Context
============================================
Minimal windowing module for real-time playback.

Responsibilities:
- Create desktop window (GLFW)
- Initialize OpenGL 4.5+ core context
- Own main render loop
- Handle resize and close events

This module does NOT render anything beyond clearing the screen.
"""

import glfw
from OpenGL.GL import *
import sys
from typing import Optional, Tuple


class PlaybackWindow:
    """
    Desktop window with OpenGL 4.5+ context for real-time playback.
    
    Owns the main render loop and handles window events.
    """
    
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "KizunaSR Player",
        vsync: bool = True
    ):
        """
        Initialize playback window.
        
        Args:
            width: Initial window width
            height: Initial window height
            title: Window title
            vsync: Enable V-sync (swap interval = 1)
        
        Raises:
            RuntimeError: If GLFW or OpenGL context creation fails
        """
        self.width = width
        self.height = height
        self.title = title
        self.vsync = vsync
        
        self.window: Optional[glfw._GLFWwindow] = None
        self.framebuffer_width = width
        self.framebuffer_height = height
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        print("[Window] GLFW initialized")
        
        # Configure OpenGL context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)  # macOS compatibility
        glfw.window_hint(glfw.RESIZABLE, GL_TRUE)
        
        # Create window
        self.window = glfw.create_window(width, height, title, None, None)
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        print(f"[Window] Created: {width}×{height}")
        
        # Make context current
        glfw.make_context_current(self.window)
        
        # Enable V-sync
        if vsync:
            glfw.swap_interval(1)
            print("[Window] V-sync enabled")
        else:
            glfw.swap_interval(0)
        
        # Verify OpenGL version
        gl_version = glGetString(GL_VERSION).decode('utf-8')
        gl_renderer = glGetString(GL_RENDERER).decode('utf-8')
        
        print(f"[OpenGL] Version: {gl_version}")
        print(f"[OpenGL] Renderer: {gl_renderer}")
        
        # Check if we have OpenGL 4.5+
        major = glGetIntegerv(GL_MAJOR_VERSION)
        minor = glGetIntegerv(GL_MINOR_VERSION)
        
        if major < 4 or (major == 4 and minor < 5):
            print(f"[Warning] OpenGL {major}.{minor} detected, 4.5+ recommended")
        else:
            print(f"[OpenGL] Version {major}.{minor} confirmed")
        
        # Set callbacks
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        
        # Initialize framebuffer size
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        self.framebuffer_width = fb_w
        self.framebuffer_height = fb_h
        
        # Set initial OpenGL state
        glViewport(0, 0, fb_w, fb_h)
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Black clear color
        
        print(f"[Window] Framebuffer size: {fb_w}×{fb_h}")
        print("[Window] Initialization complete")
    
    def _framebuffer_size_callback(self, window, width: int, height: int):
        """
        Callback for framebuffer resize events.
        
        Args:
            window: GLFW window handle
            width: New framebuffer width
            height: New framebuffer height
        """
        self.framebuffer_width = width
        self.framebuffer_height = height
        
        # Update OpenGL viewport
        glViewport(0, 0, width, height)
        
        print(f"[Window] Framebuffer resized: {width}×{height}")
    
    def _key_callback(self, window, key: int, scancode: int, action: int, mods: int):
        """
        Callback for keyboard events.
        
        Args:
            window: GLFW window handle
            key: Key code
            scancode: Platform-specific scancode
            action: GLFW_PRESS, GLFW_RELEASE, or GLFW_REPEAT
            mods: Modifier keys (shift, ctrl, etc.)
        """
        # ESC to close window
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
            print("[Window] ESC pressed, closing...")
    
    def should_close(self) -> bool:
        """
        Check if window should close.
        
        Returns:
            True if window close was requested
        """
        return glfw.window_should_close(self.window)
    
    def get_framebuffer_size(self) -> Tuple[int, int]:
        """
        Get current framebuffer size.
        
        Returns:
            (width, height) tuple
        """
        return (self.framebuffer_width, self.framebuffer_height)
    
    def poll_events(self):
        """Poll window events (keyboard, mouse, resize, etc.)."""
        glfw.poll_events()
    
    def swap_buffers(self):
        """Swap front and back buffers (present frame)."""
        glfw.swap_buffers(self.window)
    
    def run(self, frame_callback=None):
        """
        Run the main render loop.
        
        Args:
            frame_callback: Optional callback function called each frame
                            Signature: callback() -> None
        
        The render loop:
        1. Polls events
        2. Calls frame_callback (if provided)
        3. Swaps buffers
        4. Repeats until window close
        """
        print("[Window] Entering main loop")
        
        frame_count = 0
        
        while not self.should_close():
            # Poll events (keyboard, resize, etc.)
            self.poll_events()
            
            # Clear framebuffer (default rendering)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # User-provided per-frame callback
            if frame_callback:
                frame_callback()
            
            # Present frame
            self.swap_buffers()
            
            frame_count += 1
            
            # Log every 1000 frames
            if frame_count % 1000 == 0:
                print(f"[Window] Frame {frame_count}")
        
        print(f"[Window] Exiting main loop (total frames: {frame_count})")
    
    def close(self):
        """Clean up and destroy window."""
        if self.window:
            glfw.destroy_window(self.window)
            self.window = None
            print("[Window] Window destroyed")
        
        glfw.terminate()
        print("[Window] GLFW terminated")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor: ensure cleanup."""
        self.close()


def main():
    """Example usage: Basic window with empty render loop."""
    
    print("=" * 60)
    print("KizunaSR Playback Window - Demo")
    print("=" * 60)
    print("Press ESC to close window")
    print("=" * 60)
    
    try:
        # Create window with OpenGL context
        with PlaybackWindow(width=1280, height=720, title="KizunaSR Player - Demo") as window:
            
            # Define a simple frame callback (optional)
            frame_count = [0]  # Mutable for closure
            
            def on_frame():
                """Called every frame."""
                frame_count[0] += 1
                
                # Just clear to a color (demo)
                # Actual rendering would happen here
                pass
            
            # Run main loop
            window.run(frame_callback=on_frame)
            
            print(f"[Demo] Rendered {frame_count[0]} frames")
    
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("[Demo] Clean shutdown")


if __name__ == "__main__":
    main()

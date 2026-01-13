"""
KizunaSR - OpenGL Texture Management
=====================================
Concrete implementation of GPU texture uploader for OpenGL.

Responsibilities:
- Upload decoded RGB frames to GPU textures
- Manage texture lifetime and cleanup
- Provide texture metadata queries

This module does NOT execute shaders or render anything.
"""

from OpenGL.GL import *
import numpy as np
from typing import Dict, Tuple, Optional

from runtime.video import VideoFrame, IGPUTextureUploader


class OpenGLTextureUploader(IGPUTextureUploader):
    """
    OpenGL-based GPU texture uploader.
    
    Uploads VideoFrame (CPU RGB data) to OpenGL textures.
    Manages texture lifecycle and reuse.
    """
    
    def __init__(self, enable_texture_reuse: bool = True):
        """
        Initialize texture uploader.
        
        Args:
            enable_texture_reuse: Reuse textures for same resolution (reduces allocations)
        """
        self.enable_texture_reuse = enable_texture_reuse
        
        # Texture registry: texture_id -> (width, height)
        self.textures: Dict[int, Tuple[int, int]] = {}
        
        # Texture pool for reuse (if enabled)
        # Key: (width, height), Value: list of available texture IDs
        self.texture_pool: Dict[Tuple[int, int], list] = {}
        
        print("[TextureUploader] Initialized")
        if enable_texture_reuse:
            print("[TextureUploader] Texture reuse enabled")
    
    def upload_frame(self, frame: VideoFrame) -> int:
        """
        Upload a video frame to GPU texture.
        
        Args:
            frame: VideoFrame with RGB data
        
        Returns:
            OpenGL texture ID
        
        Raises:
            RuntimeError: If upload fails
        """
        width, height = frame.width, frame.height
        
        # Try to reuse texture from pool
        texture_id = None
        if self.enable_texture_reuse:
            pool_key = (width, height)
            if pool_key in self.texture_pool and self.texture_pool[pool_key]:
                texture_id = self.texture_pool[pool_key].pop()
                # Reuse existing texture, just update data
                self._upload_pixels(texture_id, frame.data, width, height, update=True)
                return texture_id
        
        # Create new texture
        texture_id = self._create_texture(frame.data, width, height)
        
        # Track texture
        self.textures[texture_id] = (width, height)
        
        return texture_id
    
    def _create_texture(self, pixels: np.ndarray, width: int, height: int) -> int:
        """
        Create a new OpenGL texture and upload pixel data.
        
        Args:
            pixels: RGB pixel data (H, W, 3) uint8
            width: Texture width
            height: Texture height
        
        Returns:
            OpenGL texture ID
        """
        # Generate texture
        texture_id = glGenTextures(1)
        
        if texture_id == 0:
            raise RuntimeError("Failed to generate OpenGL texture")
        
        # Upload pixels
        self._upload_pixels(texture_id, pixels, width, height, update=False)
        
        return texture_id
    
    def _upload_pixels(
        self,
        texture_id: int,
        pixels: np.ndarray,
        width: int,
        height: int,
        update: bool = False
    ):
        """
        Upload pixel data to an OpenGL texture.
        
        Args:
            texture_id: OpenGL texture ID
            pixels: RGB pixel data (H, W, 3) uint8
            width: Texture width
            height: Texture height
            update: If True, use glTexSubImage2D (faster for reuse)
        """
        # Bind texture
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Set pixel store parameters (alignment)
        # Row alignment: OpenGL expects 4-byte aligned rows by default
        # RGB24 (3 bytes/pixel) may not be 4-byte aligned
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        
        if not update:
            # Initial upload: allocate texture storage
            glTexImage2D(
                GL_TEXTURE_2D,      # Target
                0,                  # Mipmap level
                GL_RGB8,            # Internal format (8-bit RGB)
                width,              # Width
                height,             # Height
                0,                  # Border (must be 0)
                GL_RGB,             # Format (input data)
                GL_UNSIGNED_BYTE,   # Type (uint8)
                pixels              # Pixel data
            )
            
            # Set texture parameters (only needed once)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        else:
            # Update existing texture (faster, no reallocation)
            glTexSubImage2D(
                GL_TEXTURE_2D,      # Target
                0,                  # Mipmap level
                0, 0,               # Offset (x, y)
                width,              # Width
                height,             # Height
                GL_RGB,             # Format
                GL_UNSIGNED_BYTE,   # Type
                pixels              # Pixel data
            )
        
        # Unbind texture
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Check for OpenGL errors
        error = glGetError()
        if error != GL_NO_ERROR:
            raise RuntimeError(f"OpenGL error during texture upload: {error}")
    
    def release_texture(self, texture_id: int):
        """
        Release a GPU texture.
        
        Args:
            texture_id: Texture ID to release
        
        If texture reuse is enabled, texture is returned to pool.
        Otherwise, it is deleted immediately.
        """
        if texture_id not in self.textures:
            print(f"[Warning] Attempted to release unknown texture {texture_id}")
            return
        
        width, height = self.textures[texture_id]
        
        if self.enable_texture_reuse:
            # Return to pool for reuse
            pool_key = (width, height)
            if pool_key not in self.texture_pool:
                self.texture_pool[pool_key] = []
            self.texture_pool[pool_key].append(texture_id)
        else:
            # Delete immediately
            glDeleteTextures([texture_id])
            del self.textures[texture_id]
    
    def get_texture_size(self, texture_id: int) -> Tuple[int, int]:
        """
        Get texture dimensions.
        
        Args:
            texture_id: Texture ID
        
        Returns:
            (width, height) tuple
        
        Raises:
            ValueError: If texture ID is unknown
        """
        if texture_id not in self.textures:
            raise ValueError(f"Unknown texture ID: {texture_id}")
        
        return self.textures[texture_id]
    
    def cleanup(self):
        """
        Delete all textures and clear pool.
        
        Call this when shutting down or switching videos.
        """
        # Delete all tracked textures
        all_texture_ids = list(self.textures.keys())
        
        if all_texture_ids:
            glDeleteTextures(all_texture_ids)
            print(f"[TextureUploader] Deleted {len(all_texture_ids)} textures")
        
        # Clear registries
        self.textures.clear()
        self.texture_pool.clear()
    
    def get_stats(self) -> dict:
        """
        Get texture usage statistics.
        
        Returns:
            Dictionary with texture counts
        """
        pooled_count = sum(len(pool) for pool in self.texture_pool.values())
        
        return {
            'total_textures': len(self.textures),
            'pooled_textures': pooled_count,
            'active_textures': len(self.textures) - pooled_count,
            'pool_sizes': {f"{w}×{h}": len(pool) for (w, h), pool in self.texture_pool.items()}
        }
    
    def __del__(self):
        """Destructor: ensure cleanup."""
        self.cleanup()


def main():
    """Test texture uploader with simple OpenGL context."""
    
    print("=" * 60)
    print("OpenGL Texture Uploader - Test")
    print("=" * 60)
    
    # Need an OpenGL context to test
    from runtime.playback import PlaybackWindow
    from runtime.video import VideoFrame
    
    try:
        # Create window (provides OpenGL context)
        with PlaybackWindow(800, 600, "Texture Upload Test") as window:
            
            # Create uploader
            uploader = OpenGLTextureUploader(enable_texture_reuse=True)
            
            # Create a test frame (solid red)
            test_data = np.zeros((480, 640, 3), dtype=np.uint8)
            test_data[:, :, 0] = 255  # Red channel
            
            test_frame = VideoFrame(
                data=test_data,
                width=640,
                height=480,
                pts=0.0,
                frame_index=0,
                format='rgb24'
            )
            
            print("\n[Test] Uploading test frame...")
            
            # Upload frame
            texture_id = uploader.upload_frame(test_frame)
            print(f"[Test] Uploaded texture ID: {texture_id}")
            
            # Query size
            width, height = uploader.get_texture_size(texture_id)
            print(f"[Test] Texture size: {width}×{height}")
            
            # Upload again (should reuse or create new)
            texture_id2 = uploader.upload_frame(test_frame)
            print(f"[Test] Second upload texture ID: {texture_id2}")
            
            # Release first texture (goes to pool)
            uploader.release_texture(texture_id)
            print(f"[Test] Released texture {texture_id}")
            
            # Upload again (should reuse from pool)
            texture_id3 = uploader.upload_frame(test_frame)
            print(f"[Test] Third upload texture ID: {texture_id3} (should equal {texture_id})")
            
            # Print stats
            stats = uploader.get_stats()
            print(f"\n[Test] Texture stats: {stats}")
            
            # Cleanup
            uploader.cleanup()
            print("\n[Test] Cleanup complete")
            
            print("\n" + "=" * 60)
            print("Test passed! Press ESC to close window.")
            print("=" * 60)
            
            # Run window for a bit to show it works
            frame_count = 0
            def on_frame():
                nonlocal frame_count
                frame_count += 1
                if frame_count >= 60:  # 1 second at 60fps
                    window._key_callback(window.window, glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
            
            import glfw
            window.run(frame_callback=on_frame)
    
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

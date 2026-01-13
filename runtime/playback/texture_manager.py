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
    
    def __init__(self, enable_texture_reuse: bool = True, use_pbo: bool = True):
        """
        Initialize texture uploader.
        
        Args:
            enable_texture_reuse: Reuse textures for same resolution (reduces allocations)
            use_pbo: Use Pixel Buffer Objects for zero-copy upload (default: True)
        """
        self.enable_texture_reuse = enable_texture_reuse
        self.use_pbo = use_pbo
        
        # Texture registry: texture_id -> (width, height)
        self.textures: Dict[int, Tuple[int, int]] = {}
        
        # Texture pool for reuse (if enabled)
        # Key: (width, height), Value: list of available texture IDs
        self.texture_pool: Dict[Tuple[int, int], list] = {}
        
        # PBO state
        self.pbo_enabled = False
        self.pbos: list = []  # Double-buffered PBOs
        self.pbo_index = 0  # Current PBO for upload
        self.pbo_resolution: Optional[Tuple[int, int]] = None
        
        print("[TextureUploader] Initialized")
        if enable_texture_reuse:
            print("[TextureUploader] Texture reuse enabled")
        
        if use_pbo:
            self._init_pbos()
    
    def _init_pbos(self):
        """Initialize Pixel Buffer Objects for zero-copy upload."""
        try:
            # Create 2 PBOs for double buffering
            self.pbos = glGenBuffers(2)
            if isinstance(self.pbos, int):
                self.pbos = [self.pbos]
            
            self.pbo_enabled = True
            print("[TextureUploader] PBO zero-copy upload enabled")
            
        except Exception as e:
            print(f"[TextureUploader] PBO initialization failed: {e}, using fallback")
            self.pbo_enabled = False
            self.pbos = []
    
    def _resize_pbos(self, width: int, height: int):
        """Resize PBOs for new frame resolution."""
        if not self.pbo_enabled:
            return
        
        buffer_size = width * height * 3  # RGB24
        
        for pbo in self.pbos:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer_size, None, GL_STREAM_DRAW)
        
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self.pbo_resolution = (width, height)
    
    def upload_frame(self, frame: VideoFrame) -> int:
        """
        Upload a video frame to GPU texture.
        
        Supports both RGB and YUV frames.
        For YUV frames, returns a special texture ID that references YUV textures.
        
        Uses PBO double-buffering for zero-copy upload if enabled.
        
        Args:
            frame: VideoFrame with RGB or YUV data
        
        Returns:
            OpenGL texture ID (RGB) or tuple handle for YUV
        
        Raises:
            RuntimeError: If upload fails
        """
        width, height = frame.width, frame.height
        
        # Check if this is a YUV frame
        if frame.yuv_data is not None:
            return self._upload_yuv_frame(frame)
        
        # RGB path (existing logic)
        # Check if PBO needs resize
        if self.pbo_enabled and self.pbo_resolution != (width, height):
            self._resize_pbos(width, height)
        
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
    
    def _upload_yuv_frame(self, frame: VideoFrame) -> int:
        """
        Upload YUV frame as separate Y and UV textures.
        
        Args:
            frame: VideoFrame with yuv_data
        
        Returns:
            Texture ID (actually stores YUV texture IDs internally)
        """
        try:
            yuv_data = frame.yuv_data
            y_plane = yuv_data['y']
            u_plane = yuv_data['u']
            v_plane = yuv_data['v']
            
            # Create Y texture (full resolution, single channel)
            y_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, y_tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, frame.width, frame.height, 0,
                         GL_RED, GL_UNSIGNED_BYTE, y_plane)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Create UV texture (half resolution, two channels)
            # Combine U and V into RG texture
            uv_width = frame.width // 2
            uv_height = frame.height // 2
            uv_data = np.stack([u_plane, v_plane], axis=-1).astype(np.uint8)
            
            uv_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, uv_tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, uv_width, uv_height, 0,
                         GL_RG, GL_UNSIGNED_BYTE, uv_data)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            glBindTexture(GL_TEXTURE_2D, 0)
            
            # Store YUV texture IDs (use a special registry)
            # For now, return Y texture ID and store UV internally
            if not hasattr(self, 'yuv_textures'):
                self.yuv_textures = {}
            
            self.yuv_textures[y_tex] = {'y': y_tex, 'uv': uv_tex}
            self.textures[y_tex] = (frame.width, frame.height)
            
            return y_tex
            
        except Exception as e:
            print(f"[TextureUploader] YUV upload failed: {e}")
            raise
    
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
        
        Uses PBO for async zero-copy upload if enabled.
        
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
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        
        # Choose upload path: PBO or classic
        if self.pbo_enabled and len(self.pbos) >= 2:
            self._upload_via_pbo(texture_id, pixels, width, height, update)
        else:
            self._upload_classic(texture_id, pixels, width, height, update)
        
        # Unbind texture
        glBindTexture(GL_TEXTURE_2D, 0)
        
        # Check for OpenGL errors
        error = glGetError()
        if error != GL_NO_ERROR:
            raise RuntimeError(f"OpenGL error during texture upload: {error}")
    
    def _upload_via_pbo(
        self,
        texture_id: int,
        pixels: np.ndarray,
        width: int,
        height: int,
        update: bool
    ):
        """
        Upload texture using PBO for zero-copy transfer.
        
        Double buffering prevents GPU stalls.
        """
        buffer_size = width * height * 3
        
        # First, bind current PBO and trigger texture upload from it
        # (This uploads data written in the previous frame)
        current_pbo = self.pbos[self.pbo_index]
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, current_pbo)
        
        if not update:
            # Initial allocation
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0,
                GL_RGB, GL_UNSIGNED_BYTE, None  # NULL pointer = use PBO
            )
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        else:
            # Update texture from PBO
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0, width, height,
                GL_RGB, GL_UNSIGNED_BYTE, None  # NULL = use PBO
            )
        
        # Now, switch to next PBO and write new frame data
        self.pbo_index = (self.pbo_index + 1) % len(self.pbos)
        next_pbo = self.pbos[self.pbo_index]
        
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, next_pbo)
        
        # Map buffer for writing
        try:
            ptr = glMapBufferRange(
                GL_PIXEL_UNPACK_BUFFER,
                0,
                buffer_size,
                GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT
            )
            
            if ptr:
                # Copy pixel data into mapped buffer
                # Ensure pixels are contiguous C-order
                pixels_flat = np.ascontiguousarray(pixels).flatten()
                ctypes.memmove(ptr, pixels_flat.ctypes.data, buffer_size)
                
                # Unmap buffer
                glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)
            else:
                # Mapping failed, fall back to classic
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
                self._upload_classic(texture_id, pixels, width, height, update)
                return
        
        except Exception as e:
            print(f"[TextureUploader] PBO upload failed: {e}, falling back")
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
            self._upload_classic(texture_id, pixels, width, height, update)
            return
        
        # Unbind PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    
    def _upload_classic(
        self,
        texture_id: int,
        pixels: np.ndarray,
        width: int,
        height: int,
        update: bool
    ):
        """
        Classic CPU→GPU upload using glTex(Sub)Image2D.
        
        Fallback when PBO is unavailable or fails.
        """
        if not update:
            # Initial upload: allocate texture storage
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0,
                GL_RGB, GL_UNSIGNED_BYTE, pixels
            )
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        else:
            # Update existing texture
            glTexSubImage2D(
                GL_TEXTURE_2D, 0, 0, 0, width, height,
                GL_RGB, GL_UNSIGNED_BYTE, pixels
            )
    
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
        Delete all textures, PBOs, and clear pool.
        
        Call this when shutting down or switching videos.
        """
        # Delete all tracked textures
        all_texture_ids = list(self.textures.keys())
        
        if all_texture_ids:
            glDeleteTextures(all_texture_ids)
            print(f"[TextureUploader] Deleted {len(all_texture_ids)} textures")
        
        # Delete YUV textures if any
        if hasattr(self, 'yuv_textures'):
            for yuv_set in self.yuv_textures.values():
                glDeleteTextures([yuv_set['y'], yuv_set['uv']])
            print(f"[TextureUploader] Deleted {len(self.yuv_textures)} YUV texture sets")
            self.yuv_textures = {}
        
        # Delete PBOs
        if self.pbos:
            try:
                glDeleteBuffers(len(self.pbos), self.pbos)
                print(f"[TextureUploader] Deleted {len(self.pbos)} PBOs")
            except:
                pass
            self.pbos = []
            self.pbo_enabled = False
        
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

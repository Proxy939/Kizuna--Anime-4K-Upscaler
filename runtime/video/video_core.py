"""
KizunaSR - Shared Video Core
=============================
Mode-agnostic video decoding and frame management.

This module handles:
- Video decoding (FFmpeg via PyAV)
- Frame extraction and timing
- Frame lifecycle management
- Abstract GPU texture upload interface

It does NOT handle:
- Video encoding
- Display/windowing
- AI inference
- Shader processing
"""

import av
import numpy as np
from typing import Optional, Iterator, Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock


@dataclass
class VideoFrame:
    """
    Represents a single decoded video frame.
    
    Attributes:
        data: Frame pixel data (H, W, 3) RGB uint8
        width: Frame width in pixels
        height: Frame height in pixels
        pts: Presentation timestamp (in seconds)
        frame_index: Sequential frame number (0-indexed)
        format: Color format ('rgb24', 'yuv420p', etc.)
    """
    data: np.ndarray
    width: int
    height: int
    pts: float
    frame_index: int
    format: str = 'rgb24'
    
    def __repr__(self) -> str:
        return (f"VideoFrame(index={self.frame_index}, "
                f"size={self.width}×{self.height}, "
                f"pts={self.pts:.3f}s, format={self.format})")


class VideoDecoder:
    """
    FFmpeg-based video decoder using PyAV.
    
    Decodes video files to raw RGB frames with timing information.
    """
    
    def __init__(self, video_path: str, use_hwaccel: bool = True):
        """
        Initialize video decoder.
        
        Args:
            video_path: Path to video file
            use_hwaccel: Attempt hardware-accelerated decoding (default: True)
        
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If container cannot be opened
        """
        self.video_path = Path(video_path)
        self.use_hwaccel = use_hwaccel
        self.hwaccel_backend = None
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Try hardware decode first if requested
        if use_hwaccel:
            hwaccel_success = self._try_hwaccel_open()
            if not hwaccel_success:
                print("[VideoDecoder] Hardware decode failed, falling back to CPU")
                self._open_cpu()
        else:
            self._open_cpu()
        
        # Get video stream
        self.stream = self.container.streams.video[0]
        
        if self.stream is None:
            raise ValueError("No video stream found in file")
        
        # Extract metadata
        self.width = self.stream.width
        self.height = self.stream.height
        self.fps = float(self.stream.average_rate)
        self.duration = float(self.stream.duration * self.stream.time_base) if self.stream.duration else None
        self.total_frames = self.stream.frames if self.stream.frames else None
        self.codec = self.stream.codec_context.name
        
        # Frame index counter
        self.frame_index = 0
        
        print(f"[VideoDecoder] Opened: {self.video_path.name}")
        print(f"[VideoDecoder] Resolution: {self.width}×{self.height}")
        print(f"[VideoDecoder] FPS: {self.fps:.2f}")
        print(f"[VideoDecoder] Codec: {self.codec}")
        print(f"[VideoDecoder] Decode: {self.hwaccel_backend or 'CPU'}")
        if self.duration:
            print(f"[VideoDecoder] Duration: {self.duration:.2f}s")
        if self.total_frames:
            print(f"[VideoDecoder] Total frames: {self.total_frames}")
    
    def _detect_hwaccel_backend(self) -> Optional[str]:
        """
        Detect available hardware acceleration backend.
        
        Returns:
            Backend name or None if no hardware accel available
        """
        # Try backends in preference order
        backends = [
            'cuda',      # NVIDIA NVDEC
            'd3d11va',   # Windows Direct3D 11
            'dxva2',     # Windows DirectX Video Acceleration
            'vaapi',     # Linux VA-API
            'videotoolbox'  # macOS VideoToolbox
        ]
        
        for backend in backends:
            if backend in av.codec.hwaccels:
                return backend
        
        return None
    
    def _try_hwaccel_open(self) -> bool:
        """
        Attempt to open video with hardware acceleration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Detect available backend
            backend = self._detect_hwaccel_backend()
            
            if backend is None:
                print("[VideoDecoder] No hardware decode backends available")
                return False
            
            # Open container with hwaccel
            options = {'hwaccel': backend}
            
            try:
                self.container = av.open(str(self.video_path), options=options)
                self.hwaccel_backend = backend.upper()
                print(f"[VideoDecoder] Using {self.hwaccel_backend} hardware decode")
                return True
                
            except av.AVError as e:
                print(f"[VideoDecoder] {backend} init failed: {e}")
                return False
        
        except Exception as e:
            print(f"[VideoDecoder] Hardware decode setup error: {e}")
            return False
    
    def _open_cpu(self):
        """
        Open video with CPU decoding (fallback).
        
        Raises:
            ValueError: If container cannot be opened
        """
        try:
            self.container = av.open(str(self.video_path))
            self.hwaccel_backend = None
        except av.AVError as e:
            raise ValueError(f"Failed to open video: {e}")
    
    def __iter__(self) -> Iterator[VideoFrame]:
        """
        Iterate through all frames in the video.
        
        Yields:
            VideoFrame objects in presentation order
        """
        self.frame_index = 0
        
        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                yield self._convert_frame(frame)
    
    def _convert_frame(self, av_frame: av.VideoFrame) -> VideoFrame:
        """
        Convert PyAV frame to VideoFrame.
        
        Args:
            av_frame: PyAV video frame
        
        Returns:
            VideoFrame with RGB data
        """
        # Convert to RGB24
        rgb_frame = av_frame.to_rgb().to_ndarray()
        
        # Calculate PTS in seconds
        pts = float(av_frame.pts * self.stream.time_base) if av_frame.pts else 0.0
        
        # Create VideoFrame
        video_frame = VideoFrame(
            data=rgb_frame,
            width=av_frame.width,
            height=av_frame.height,
            pts=pts,
            frame_index=self.frame_index,
            format='rgb24'
        )
        
        self.frame_index += 1
        
        return video_frame
    
    def seek(self, timestamp: float):
        """
        Seek to a specific timestamp.
        
        Args:
            timestamp: Time in seconds
        """
        # Convert timestamp to stream time base
        pts = int(timestamp / self.stream.time_base)
        self.container.seek(pts, stream=self.stream)
        self.frame_index = int(timestamp * self.fps)
    
    def close(self):
        """Close the video container and release resources."""
        if hasattr(self, 'container') and self.container:
            self.container.close()
            print(f"[VideoDecoder] Closed: {self.video_path.name}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor: ensure resources are freed."""
        self.close()


class FrameQueue:
    """
    Thread-safe frame buffer for producer-consumer pattern.
    
    Manages frame lifecycle: acquire → process → release
    """
    
    def __init__(self, maxsize: int = 16):
        """
        Initialize frame queue.
        
        Args:
            maxsize: Maximum queue size (bounds memory usage)
        """
        self.queue: Queue[VideoFrame] = Queue(maxsize=maxsize)
        self.lock = Lock()
        self._total_enqueued = 0
        self._total_dequeued = 0
        
        print(f"[FrameQueue] Initialized with maxsize={maxsize}")
    
    def enqueue(self, frame: VideoFrame, block: bool = True, timeout: Optional[float] = None):
        """
        Add a frame to the queue.
        
        Args:
            frame: VideoFrame to enqueue
            block: Whether to block if queue is full
            timeout: Timeout in seconds (None = infinite)
        
        Raises:
            queue.Full: If queue is full and block=False
        """
        self.queue.put(frame, block=block, timeout=timeout)
        
        with self.lock:
            self._total_enqueued += 1
        
        if self._total_enqueued % 100 == 0:
            print(f"[FrameQueue] Enqueued {self._total_enqueued} frames (queue size: {self.size()})")
    
    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        """
        Remove and return a frame from the queue.
        
        Args:
            block: Whether to block if queue is empty
            timeout: Timeout in seconds (None = infinite)
        
        Returns:
            VideoFrame or None if queue is empty and block=False
        
        Raises:
            queue.Empty: If queue is empty and block=False
        """
        try:
            frame = self.queue.get(block=block, timeout=timeout)
            
            with self.lock:
                self._total_dequeued += 1
            
            return frame
        except:
            return None
    
    def size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.queue.full()
    
    def stats(self) -> dict:
        """Get queue statistics."""
        with self.lock:
            return {
                'current_size': self.size(),
                'total_enqueued': self._total_enqueued,
                'total_dequeued': self._total_dequeued,
                'pending': self._total_enqueued - self._total_dequeued
            }


# ==============================================================================
# Abstract GPU Upload Interface
# ==============================================================================

class IGPUTextureUploader:
    """
    Abstract interface for uploading frames to GPU textures.
    
    Different graphics APIs (OpenGL, Vulkan, WebGPU) will implement this.
    """
    
    def upload_frame(self, frame: VideoFrame) -> int:
        """
        Upload a frame to GPU texture.
        
        Args:
            frame: VideoFrame to upload
        
        Returns:
            Texture handle/ID (API-specific)
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement upload_frame()")
    
    def release_texture(self, texture_id: int):
        """
        Release GPU texture resources.
        
        Args:
            texture_id: Texture handle to release
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement release_texture()")
    
    def get_texture_size(self, texture_id: int) -> tuple[int, int]:
        """
        Get texture dimensions.
        
        Args:
            texture_id: Texture handle
        
        Returns:
            (width, height) tuple
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_texture_size()")


# Example stub implementations (to be replaced by actual graphics API code)

class OpenGLTextureUploader(IGPUTextureUploader):
    """Placeholder for OpenGL implementation."""
    
    def upload_frame(self, frame: VideoFrame) -> int:
        # TODO: Implement OpenGL glTexImage2D upload
        print(f"[OpenGL] Would upload frame {frame.frame_index} ({frame.width}×{frame.height})")
        return frame.frame_index  # Stub: return frame index as texture ID
    
    def release_texture(self, texture_id: int):
        # TODO: Implement OpenGL glDeleteTextures
        print(f"[OpenGL] Would release texture {texture_id}")
    
    def get_texture_size(self, texture_id: int) -> tuple[int, int]:
        # TODO: Implement OpenGL glGetTexLevelParameter
        return (0, 0)  # Stub


class VulkanTextureUploader(IGPUTextureUploader):
    """Placeholder for Vulkan implementation."""
    
    def upload_frame(self, frame: VideoFrame) -> int:
        # TODO: Implement Vulkan vkCmdCopyBufferToImage
        print(f"[Vulkan] Would upload frame {frame.frame_index} ({frame.width}×{frame.height})")
        return frame.frame_index
    
    def release_texture(self, texture_id: int):
        # TODO: Implement Vulkan vkDestroyImage
        print(f"[Vulkan] Would release texture {texture_id}")
    
    def get_texture_size(self, texture_id: int) -> tuple[int, int]:
        return (0, 0)


# ==============================================================================
# Frame Processor Orchestrator
# ==============================================================================

class VideoFrameProcessor:
    """
    Orchestrates the full frame lifecycle:
    Decode → Queue → Upload → Process → Release
    
    This is mode-agnostic: it doesn't decide whether frames
    are displayed or encoded.
    """
    
    def __init__(
        self,
        video_path: str,
        gpu_uploader: Optional[IGPUTextureUploader] = None,
        queue_size: int = 16
    ):
        """
        Initialize frame processor.
        
        Args:
            video_path: Path to video file
            gpu_uploader: GPU texture uploader (None = CPU-only mode)
            queue_size: Maximum frame queue size
        """
        self.decoder = VideoDecoder(video_path)
        self.queue = FrameQueue(maxsize=queue_size)
        self.gpu_uploader = gpu_uploader
        
        print(f"[FrameProcessor] Initialized")
        print(f"[FrameProcessor] GPU upload: {'Enabled' if gpu_uploader else 'Disabled (CPU-only)'}")
    
    def process_all_frames(
        self,
        frame_callback: Callable[[VideoFrame, Optional[int]], None]
    ):
        """
        Process all frames in the video.
        
        Args:
            frame_callback: Function called for each frame
                            Signature: (frame, texture_id) -> None
        """
        print(f"[FrameProcessor] Starting frame processing")
        
        for frame in self.decoder:
            # Enqueue frame
            self.queue.enqueue(frame)
            
            # Dequeue immediately (simple sequential processing)
            queued_frame = self.queue.dequeue()
            
            if queued_frame is None:
                print(f"[Warning] Failed to dequeue frame {frame.frame_index}")
                continue
            
            # Upload to GPU (if uploader available)
            texture_id = None
            if self.gpu_uploader:
                try:
                    texture_id = self.gpu_uploader.upload_frame(queued_frame)
                except Exception as e:
                    print(f"[Error] GPU upload failed for frame {queued_frame.frame_index}: {e}")
            
            # Call user-provided callback
            frame_callback(queued_frame, texture_id)
            
            # Release GPU texture (if uploaded)
            if texture_id is not None and self.gpu_uploader:
                try:
                    self.gpu_uploader.release_texture(texture_id)
                except Exception as e:
                    print(f"[Error] Texture release failed: {e}")
        
        # Print final stats
        stats = self.queue.stats()
        print(f"[FrameProcessor] Processing complete")
        print(f"[FrameProcessor] Total frames: {stats['total_dequeued']}")
    
    def close(self):
        """Clean up resources."""
        self.decoder.close()


def main():
    """Example usage."""
    
    video_path = "test_video.mp4"
    
    if not Path(video_path).exists():
        print(f"[Info] {video_path} not found, skipping example")
        return
    
    # Example 1: Decode and print frame info (CPU-only)
    print("=" * 60)
    print("Example 1: CPU-only frame extraction")
    print("=" * 60)
    
    def frame_callback(frame: VideoFrame, texture_id: Optional[int]):
        if frame.frame_index % 30 == 0:  # Print every 30th frame
            print(f"  {frame}")
    
    processor = VideoFrameProcessor(video_path, gpu_uploader=None)
    processor.process_all_frames(frame_callback)
    processor.close()
    
    # Example 2: With GPU upload (stub)
    print("\n" + "=" * 60)
    print("Example 2: With GPU upload (stub)")
    print("=" * 60)
    
    gpu_uploader = OpenGLTextureUploader()  # Stub implementation
    
    processor_gpu = VideoFrameProcessor(video_path, gpu_uploader=gpu_uploader)
    processor_gpu.process_all_frames(frame_callback)
    processor_gpu.close()


if __name__ == "__main__":
    main()

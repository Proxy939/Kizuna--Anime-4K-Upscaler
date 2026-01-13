"""
KizunaSR - Offline Video Processor
===================================
Deterministic, high-quality video processing pipeline.

Decode → Process (Shader/AI) → Encode

This is the QUALITY PATH for KizunaSR, not real-time playback.
"""

import av
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from PIL import Image

from runtime.video import VideoDecoder, VideoFrame
from core.pipeline import KizunaSRPipeline, PipelineConfig


@dataclass
class EncoderConfig:
    """Configuration for video encoding."""
    
    # Output settings
    codec: str = 'libx264'          # Video codec (libx264, libx265, etc.)
    pixel_format: str = 'yuv420p'   # Pixel format
    
    # Quality settings
    crf: int = 18                   # Constant Rate Factor (0-51, lower=better)
    preset: str = 'medium'          # Encoding preset (ultrafast, fast, medium, slow, veryslow)
    
    # Bitrate settings (alternative to CRF)
    bitrate: Optional[str] = None   # e.g., '5M' for 5 Mbps (overrides CRF if set)
    
    # Audio settings
    audio_codec: str = 'aac'        # Audio codec
    audio_bitrate: str = '192k'     # Audio bitrate
    
    # Container format
    format: str = 'mp4'             # Output container format


@dataclass
class OfflineProcessorConfig:
    """Configuration for offline video processing."""
    
    # Input/Output
    input_path: str
    output_path: str
    
    # KizunaSR pipeline config
    pipeline_config: PipelineConfig
    
    # Encoder config
    encoder_config: EncoderConfig = None
    
    # Processing options
    audio_passthrough: bool = True   # Copy audio without re-encoding
    preserve_metadata: bool = True   # Copy metadata from input
    
    def __post_init__(self):
        if self.encoder_config is None:
            self.encoder_config = EncoderConfig()


class VideoEncoder:
    """
    FFmpeg-based video encoder using PyAV.
    
    Encodes processed frames to output video file.
    """
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        config: EncoderConfig
    ):
        """
        Initialize video encoder.
        
        Args:
            output_path: Path to output video file
            width: Output frame width
            height: Output frame height
            fps: Output frame rate
            config: Encoder configuration
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.config = config
        
        # Create output container
        self.container = av.open(str(self.output_path), 'w')
        
        # Create video stream
        self.video_stream = self.container.add_stream(config.codec, rate=fps)
        self.video_stream.width = width
        self.video_stream.height = height
        self.video_stream.pix_fmt = config.pixel_format
        
        # Set encoding options
        if config.bitrate:
            self.video_stream.bit_rate = self._parse_bitrate(config.bitrate)
        else:
            self.video_stream.options = {'crf': str(config.crf)}
        
        self.video_stream.options['preset'] = config.preset
        
        # Audio stream (will be set by add_audio_stream)
        self.audio_stream = None
        
        # Frame counter
        self.frame_count = 0
        
        print(f"[VideoEncoder] Initialized")
        print(f"[VideoEncoder] Output: {self.output_path}")
        print(f"[VideoEncoder] Resolution: {width}×{height}")
        print(f"[VideoEncoder] FPS: {fps}")
        print(f"[VideoEncoder] Codec: {config.codec}")
        print(f"[VideoEncoder] Quality: CRF {config.crf if not config.bitrate else f'bitrate {config.bitrate}'}")
    
    def _parse_bitrate(self, bitrate_str: str) -> int:
        """Convert bitrate string like '5M' to bits per second."""
        if bitrate_str.endswith('M'):
            return int(float(bitrate_str[:-1]) * 1_000_000)
        elif bitrate_str.endswith('k'):
            return int(float(bitrate_str[:-1]) * 1_000)
        else:
            return int(bitrate_str)
    
    def add_audio_stream(self, input_stream: av.audio.AudioStream, passthrough: bool = True):
        """
        Add audio stream to output.
        
        Args:
            input_stream: Input audio stream to copy
            passthrough: If True, copy without re-encoding
        """
        if passthrough:
            # Copy audio stream without re-encoding
            try:
                self.audio_stream = self.container.add_stream(template=input_stream)
                print(f"[VideoEncoder] Audio: Passthrough ({input_stream.codec.name})")
            except Exception as e:
                print(f"[Warning] Audio passthrough failed: {e}, falling back to re-encode")
                passthrough = False
        
        if not passthrough:
            # Re-encode audio
            self.audio_stream = self.container.add_stream(
                self.config.audio_codec,
                rate=input_stream.rate
            )
            self.audio_stream.bit_rate = self._parse_bitrate(self.config.audio_bitrate)
            print(f"[VideoEncoder] Audio: Re-encoding to {self.config.audio_codec} @ {self.config.audio_bitrate}")
    
    def encode_frame(self, frame_data: np.ndarray, pts: float):
        """
        Encode a single frame.
        
        Args:
            frame_data: RGB frame data (H, W, 3) uint8
            pts: Presentation timestamp in seconds
        """
        # Convert numpy array to PyAV VideoFrame
        video_frame = av.VideoFrame.from_ndarray(frame_data, format='rgb24')
        
        # Set timestamp
        video_frame.pts = int(pts / self.video_stream.time_base)
        
        # Encode frame
        for packet in self.video_stream.encode(video_frame):
            self.container.mux(packet)
        
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            print(f"[VideoEncoder] Encoded {self.frame_count} frames")
    
    def copy_audio_packet(self, packet: av.Packet):
        """
        Copy audio packet to output (passthrough mode).
        
        Args:
            packet: Audio packet to copy
        """
        if self.audio_stream:
            packet.stream = self.audio_stream
            self.container.mux(packet)
    
    def flush(self):
        """Flush encoder and finalize output."""
        # Flush video encoder
        for packet in self.video_stream.encode():
            self.container.mux(packet)
        
        print(f"[VideoEncoder] Flushed encoder")
    
    def close(self):
        """Close output container and release resources."""
        if hasattr(self, 'container') and self.container:
            self.container.close()
            print(f"[VideoEncoder] Closed: {self.output_path}")
            print(f"[VideoEncoder] Total frames encoded: {self.frame_count}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.close()


class OfflineVideoProcessor:
    """
    Offline video processing pipeline.
    
    Decode → KizunaSR Pipeline → Encode
    """
    
    def __init__(self, config: OfflineProcessorConfig):
        """
        Initialize offline processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config
        
        # Validate paths
        input_path = Path(config.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {config.input_path}")
        
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("KizunaSR Offline Video Processor")
        print("=" * 70)
        print(f"Input:  {config.input_path}")
        print(f"Output: {config.output_path}")
        print(f"Mode:   {'AI + Shader' if config.pipeline_config.use_ai else 'Shader Only'}")
        print("=" * 70)
    
    def process(self):
        """
        Execute the complete processing pipeline.
        
        Steps:
        1. Open input video (decoder)
        2. Initialize KizunaSR pipeline
        3. Initialize output encoder
        4. For each frame: decode → process → encode
        5. Handle audio passthrough
        6. Finalize output
        """
        # Open input video
        with VideoDecoder(self.config.input_path) as decoder:
            
            # Initialize KizunaSR pipeline
            print("\n[Pipeline] Initializing KizunaSR pipeline...")
            pipeline = KizunaSRPipeline(self.config.pipeline_config)
            
            # Calculate output resolution (based on shader/AI scale)
            if self.config.pipeline_config.use_ai:
                scale = self.config.pipeline_config.ai_scale
            else:
                scale = self.config.pipeline_config.shader_scale
            
            output_width = decoder.width * scale
            output_height = decoder.height * scale
            
            print(f"[Pipeline] Resolution: {decoder.width}×{decoder.height} → {output_width}×{output_height} ({scale}×)")
            
            # Initialize encoder
            with VideoEncoder(
                output_path=self.config.output_path,
                width=output_width,
                height=output_height,
                fps=decoder.fps,
                config=self.config.encoder_config
            ) as encoder:
                
                # Setup audio passthrough
                if decoder.container.streams.audio:
                    audio_stream = decoder.container.streams.audio[0]
                    encoder.add_audio_stream(audio_stream, passthrough=self.config.audio_passthrough)
                else:
                    print("[Warning] No audio stream found in input")
                
                # Process video frames
                print("\n[Processing] Starting frame processing...")
                total_frames = decoder.total_frames if decoder.total_frames else "unknown"
                
                for frame in decoder:
                    # Convert VideoFrame to PIL Image
                    input_img = Image.fromarray(frame.data, mode='RGB')
                    
                    # Run KizunaSR pipeline
                    output_img = pipeline.process_frame(input_img)
                    
                    # Convert back to numpy array
                    output_array = np.array(output_img)
                    
                    # Encode processed frame
                    encoder.encode_frame(output_array, frame.pts)
                    
                    # Progress reporting
                    if frame.frame_index % 30 == 0:
                        progress = f"{frame.frame_index}/{total_frames}" if total_frames != "unknown" else f"{frame.frame_index}"
                        print(f"[Progress] Frame {progress} ({frame.pts:.2f}s)")
                
                # Handle audio passthrough
                if self.config.audio_passthrough and encoder.audio_stream:
                    print("\n[Audio] Copying audio stream...")
                    
                    # Reopen container to read audio
                    # (decoder was already consumed for video)
                    audio_container = av.open(str(self.config.input_path))
                    audio_stream = audio_container.streams.audio[0]
                    
                    for packet in audio_container.demux(audio_stream):
                        encoder.copy_audio_packet(packet)
                    
                    audio_container.close()
                    print("[Audio] Audio copying complete")
                
                print("\n[Processing] Video processing complete")
        
        print("\n" + "=" * 70)
        print(f"SUCCESS: Output saved to {self.config.output_path}")
        print("=" * 70)


def main():
    """Example usage."""
    
    # Configure KizunaSR pipeline
    pipeline_config = PipelineConfig()
    pipeline_config.use_ai = False  # Start with shader-only for testing
    pipeline_config.shader_scale = 2
    pipeline_config.enable_temporal = False  # Disable for offline (optional)
    
    # Configure encoder
    encoder_config = EncoderConfig()
    encoder_config.codec = 'libx264'
    encoder_config.crf = 18  # High quality
    encoder_config.preset = 'medium'
    
    # Configure offline processor
    config = OfflineProcessorConfig(
        input_path="input.mp4",
        output_path="output.mp4",
        pipeline_config=pipeline_config,
        encoder_config=encoder_config,
        audio_passthrough=True
    )
    
    # Check if input exists
    if not Path(config.input_path).exists():
        print(f"[Info] {config.input_path} not found, skipping example")
        print("[Info] To test:")
        print("[Info]   1. Place a video file as 'input.mp4'")
        print("[Info]   2. Run this script")
        print("[Info]   3. Output will be saved as 'output.mp4'")
        return
    
    # Run processor
    try:
        processor = OfflineVideoProcessor(config)
        processor.process()
    except Exception as e:
        print(f"[Error] Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

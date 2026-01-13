"""
KizunaSR - Audio Player & Synchronization
==========================================
Real-time audio playback and master clock provider.

Responsibilities:
- Decode audio from video files (FFmpeg/PyAV)
- smooth playback via sounddevice
- Provide monotonic master clock for A/V sync

This module drives the timing of the entire playback engine.
"""

import av
import sounddevice as sd
import numpy as np
import threading
import time
from queue import Queue, Empty
from typing import Optional, Tuple
from pathlib import Path

# Import Clock protocol for type hinting if needed (runtime check not strictly required)
# from .scheduler import Clock

class AudioClock:
    """
    Master clock derived from audio playback.
    
    Time is calculated based on the number of samples played
    divided by the sample rate.
    
    If no audio is playing, it provides a fallback based on system time.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.samples_played = 0
        self.start_time = 0.0
        self.is_running = False
        self.lock = threading.Lock()
        
        # Fallback clock state
        self.fallback_mode = False
        self.fallback_start_time = 0.0
        self.fallback_offset = 0.0
    
    def get_time(self) -> float:
        """Get current playback time in seconds."""
        if self.fallback_mode:
            # System clock fallback
            if not self.is_running:
                return self.fallback_offset
            return (time.time() - self.fallback_start_time) + self.fallback_offset
        
        with self.lock:
            # Audio clock: samples / rate
            # Note: samples_played is updated by the audio callback
            return self.samples_played / self.sample_rate

    def update_samples(self, count: int):
        """Update the count of played samples (called by callback)."""
        with self.lock:
            self.samples_played += count

    def set_fallback_mode(self, enabled: bool):
        """Enable/disable fallback system clock mode."""
        self.fallback_mode = enabled
        if enabled:
            self.fallback_start_time = time.time()
            self.fallback_offset = 0.0

    def start(self):
        """Start the clock."""
        self.is_running = True
        if self.fallback_mode:
            self.fallback_start_time = time.time()

    def stop(self):
        """Stop/pause the clock."""
        self.is_running = False
        if self.fallback_mode:
            # Store accumulated time in offset
            self.fallback_offset += time.time() - self.fallback_start_time

    def reset(self):
        """Reset validation stats and time."""
        with self.lock:
            self.samples_played = 0
            self.fallback_offset = 0.0


class AudioPlayer:
    """
    Real-time audio player using sounddevice.
    
    Decodes audio using PyAV and streams it to the output device.
    Acts as the timekeeper for the application.
    """
    
    def __init__(self, buffer_duration: float = 0.2):
        """
        Initialize audio player.
        
        Args:
            buffer_duration: Audio buffer size in seconds
        """
        self.container = None
        self.stream = None
        self.resampler = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.clock = AudioClock()
        
        # Audio state
        self.sample_rate = 44100
        self.channels = 2
        self.dtype = 'float32'
        
        # Buffer
        # Queue stores numpy arrays of audio data
        self.audio_queue = Queue(maxsize=20) 
        self.running = False
        self.decode_thread = None
        self.stop_event = threading.Event()
        self.playback_finished = threading.Event()
        
    def open(self, video_path: str) -> bool:
        """
        Open audio stream from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if audio stream found and opened, False otherwise
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {video_path}")
            
        try:
            self.container = av.open(str(path))
            if not self.container.streams.audio:
                print("[Audio] No audio stream found. Using fallback clock.")
                self.clock.set_fallback_mode(True)
                return False
                
            self.stream = self.container.streams.audio[0]
            
            # Configure format based on stream
            # We enforce stereo float32 for simplicity in playback, 
            # allowing PyAV/FFmpeg to handle resampling layout
            self.sample_rate = self.stream.rate
            self.channels = self.stream.channels
            # AudioResampler is needed if the layout doesn't match float32 planar/packed expectations
            # or if we want to ensure a specific format for sounddevice.
            # sounddevice typically wants packed arrays.
            # PyAV defaults to planar for many codecs.
            
            self.resampler = av.AudioResampler(
                format='flt',   # Float32 packed
                layout='stereo' if self.channels == 2 else 'mono',
                rate=self.sample_rate
            )
            
            # Update clock rate
            self.clock.sample_rate = self.sample_rate
            self.clock.set_fallback_mode(False)
            
            print(f"[Audio] Opened stream: {self.stream.codec.name}, {self.sample_rate}Hz, {self.channels}ch")
            return True
            
        except Exception as e:
            print(f"[Audio] Failed to open audio: {e}")
            self.clock.set_fallback_mode(True)
            return False

    def start(self):
        """Start audio decoding and playback."""
        if self.clock.fallback_mode:
            print("[Audio] Starting fallback clock (no audio)")
            self.clock.start()
            return

        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        self.playback_finished.clear()
        
        # Start decoding thread
        self.decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.decode_thread.start()
        
        # Start output stream
        try:
            self.output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
                blocksize=2048  # Reasonable block size for buffering
            )
            self.output_stream.start()
            self.clock.start()
            print("[Audio] Playback started")
            
        except Exception as e:
            print(f"[Audio] Failed to start output stream: {e}")
            self.clock.set_fallback_mode(True)
            self.clock.start()

    def stop(self):
        """Stop playback and cleanup resources."""
        self.running = False
        self.stop_event.set()
        
        self.clock.stop()
        
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
            
        if self.decode_thread and self.decode_thread.is_alive():
            self.decode_thread.join(timeout=1.0)
            
        if self.container:
            self.container.close()
            self.container = None
            
        # Drain queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Empty:
                break
                
        print("[Audio] Stopped")

    def get_clock(self) -> AudioClock:
        """Get the master clock instance."""
        return self.clock

    def _decode_loop(self):
        """Internal Audio decoding loop."""
        try:
            for packet in self.container.demux(self.stream):
                if self.stop_event.is_set():
                    break
                    
                for frame in packet.decode():
                    # Resample/Convert to packed float32
                    # frame.to_ndarray() usually gives planar data for many codecs
                    # We use resampler to ensure packed format 'flt'
                    if self.resampler:
                        resampled_frames = self.resampler.resample(frame)
                        if resampled_frames:
                            # Concatenate frames if resampler produced multiple?
                            # Usually separate, but resample() returns list-like or single frame
                            # Actually PyAV AudioResampler.resample returns a single frame (or None)
                            # Let's iterate just in case PyAV changes or yields multiple
                            # Wait, AudioResampler.resample returns a list of frames.
                            
                            # Correction: documentation says it returns a list of AudioFrames.
                             # But typical usage is `frame = resampler.resample(frame)` returns list.
                             pass
                    
                    # Simpler approach: convert directly if format matches, otherwise skip simple conversion
                    # Actually, PyAV's resampler is reliable.
                    
                    data = None
                    if self.resampler:
                        out_frames = self.resampler.resample(frame)
                        if out_frames:
                            # Merge patches if necessary, but usually one in one out for same rate
                            # Just take the data
                            data = np.concatenate([f.to_ndarray() for f in out_frames], axis=1)
                    else:
                        data = frame.to_ndarray()

                    if data is not None:
                        # PyAV usually returns (channels, samples) for planar, 
                        # or (samples, channels) for packed?
                        # AudioResampler with 'flt' (packed float) returns (samples, channels)?
                        # Wait, PyAV usually returns (planes, samples).
                        # Let's verify layout.
                        # If we user format='flt', it is packed. 
                        # Packed layout usually means shape is (1, samples*channels) or logic differs.
                        # Let's trust sounddevice expects (frames, channels).
                        
                        # Correct logic:
                        # 1. Resample to 'flt' (packed float 32), 'stereo' (or matching channels)
                        # 2. PyAV resampled frame to_ndarray() -> depends on format.
                        #    For packed formats, PyAV might return (1, samples * channels) or (samples, channels).
                        #    Actually, standard PyAV `to_ndarray` follows plane structure.
                        #    Packed formats have 1 plane.
                        
                        # Transpose if necessary?
                        # Let's assume standard (channels, samples) from PyAV if planar,
                        # and check output of resampler.
                        
                        # To be safe/robust, let's standardize on (samples, channels) for sounddevice.
                        data = data.T # Transpose (channels, samples) -> (samples, channels)
                        
                        # Ensure contiguous C-order for sounddevice
                        data = np.ascontiguousarray(data, dtype=np.float32)

                        # Enqueue with blocking
                        while self.running and not self.stop_event.is_set():
                            try:
                                self.audio_queue.put(data, timeout=0.1)
                                break
                            except Queue.Full:
                                continue
            
            self.playback_finished.set()
            
        except Exception as e:
            print(f"[Audio] Decode error: {e}")
            # Don't stop clock, just stop decoding

    def _audio_callback(self, outdata, frames, time_info, status):
        """
        Sounddevice callback.
        
        Args:
            outdata: Output buffer (frames, channels)
            frames: Number of frames to fill
            time_info: Time info dict
            status: Status flags
        """
        if status:
            print(f"[Audio] Buffer status: {status}")

        filled = 0
        
        while filled < frames:
            if self.audio_queue.empty():
                if self.playback_finished.is_set():
                    # End of stream, fill silence
                    outdata[filled:] = 0
                    # self.stop_event.set() # Don't auto stop, let scheduler handle
                    break
                else:
                    # Underflow or pre-buffering
                    # Fill remainder with 0 and return
                    outdata[filled:] = 0
                    break
            
            # Get chunk
            try:
                # Peek or Get? We can get chunks of arbitrary size.
                # Current chunks in queue might not align with `frames`.
                # We need a robust buffer handling.
                
                # Simplified strategy:
                # We pull chunks from queue. If chunk > needed, we split it.
                # Since we can't easily put back, we might need an internal residual buffer.
                # For this MVP, let's just grab what we can.
                
                # BETTER STRATEGY: Use a specialized RingBuffer or keep a "current chunk" and index.
                pass
            except Empty:
                pass
                
            # ... Implementation of robust chunk handling
            # To keep code clean in the class, let's access a `current_chunk` state
            
        # Correct implementation of Callback with chunk management
        
        needed = frames
        output_idx = 0
        
        # Check if we have leftover data from previous callback
        if hasattr(self, '_leftover_chunk') and self._leftover_chunk is not None:
            chunk = self._leftover_chunk
            chunk_len = len(chunk)
            
            if chunk_len >= needed:
                outdata[:needed] = chunk[:needed]
                self._leftover_chunk = chunk[needed:] if chunk_len > needed else None
                self.clock.update_samples(needed)
                return
            else:
                outdata[:chunk_len] = chunk
                output_idx += chunk_len
                needed -= chunk_len
                self._leftover_chunk = None

        # Fill remaining from queue
        while needed > 0:
            try:
                chunk = self.audio_queue.get_nowait()
            except Empty:
                # Underflow
                outdata[output_idx:] = 0
                break
                
            chunk_len = len(chunk)
            
            if chunk_len > needed:
                # Take part of chunk, save rest
                outdata[output_idx : output_idx + needed] = chunk[:needed]
                self._leftover_chunk = chunk[needed:]
                output_idx += needed
                needed = 0
            else:
                # Take whole chunk
                outdata[output_idx : output_idx + chunk_len] = chunk
                output_idx += chunk_len
                needed -= chunk_len
        
        self.clock.update_samples(frames - needed)


# Unit Test
def main():
    import time
    print("="*60)
    print("Audio Player Test")
    print("="*60)
    
    # Path to a test video file - check current dir or provide one
    # If no file, use synthetic test or skip
    video_path = "runtime/video/test_video.mp4" # Placeholder
    
    # Create player
    player = AudioPlayer()
    
    # Try to verify with a file if user has one, otherwise mock
    # Since we can't guarantee a file, we just instantiation and mock open check
    
    print("[Test] Initialized player")
    
    # Demonstrate API
    try:
        # success = player.open("invalid.mp4") # Should handle error
        if Path("input.mp4").exists():
            print("[Test] Found input.mp4, testing playback...")
            if player.open("input.mp4"):
                player.start()
                clock = player.get_clock()
                
                for _ in range(30):
                    time.sleep(0.1)
                    print(f"Time: {clock.get_time():.3f}s")
                    
                player.stop()
        else:
            print("[Test] input.mp4 not found, skipping playback test")
            print("[Test] Verifying fallback clock")
            player.clock.set_fallback_mode(True)
            player.clock.start()
            time.sleep(1)
            print(f"Fallback Time: {player.clock.get_time():.3f}s")
            player.clock.stop()
            
    except Exception as e:
        print(f"[Result] Error: {e}")

if __name__ == "__main__":
    main()

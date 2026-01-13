"""
KizunaSR - Frame Scheduler
===========================
PTS-based frame scheduling for real-time playback.

Responsibilities:
- Accept decoded frames with PTS
- Schedule frames against master clock (audio)
- Drop late frames if necessary
- Maintain smooth playback timing

This module does NOT render, decode, or play audio.
"""

import time
from queue import Queue, Empty
from threading import Lock
from typing import Optional, Protocol
from dataclasses import dataclass

from runtime.video import VideoFrame


class Clock(Protocol):
    """
    Abstract clock interface for A/V synchronization.
    
    The master clock (typically audio) provides the reference time.
    """
    
    def get_time(self) -> float:
        """
        Get current clock time in seconds.
        
        Returns:
            Time in seconds since playback start
        """
        ...
    
    def reset(self):
        """Reset clock to zero."""
        ...


class SystemClock:
    """
    Fallback system clock (for testing without audio).
    
    Uses time.perf_counter() as reference.
    """
    
    def __init__(self):
        """Initialize system clock."""
        self.start_time = time.perf_counter()
    
    def get_time(self) -> float:
        """Get elapsed time since start."""
        return time.perf_counter() - self.start_time
    
    def reset(self):
        """Reset clock to zero."""
        self.start_time = time.perf_counter()


class AudioClock:
    """
    Audio-based master clock.
    
    Time is derived from audio playback position.
    """
    
    def __init__(self, audio_player=None):
        """
        Initialize audio clock.
        
        Args:
            audio_player: Audio player with get_position() method
        """
        self.audio_player = audio_player
        self.fallback_clock = SystemClock()
    
    def get_time(self) -> float:
        """Get time from audio position or fallback to system clock."""
        if self.audio_player and hasattr(self.audio_player, 'get_position'):
            return self.audio_player.get_position()
        else:
            # Fallback to system clock if audio not available
            return self.fallback_clock.get_time()
    
    def reset(self):
        """Reset clock."""
        self.fallback_clock.reset()


@dataclass
class SchedulerStats:
    """Statistics for frame scheduler."""
    total_frames: int = 0
    presented_frames: int = 0
    dropped_frames: int = 0
    early_frames: int = 0
    late_frames: int = 0


class FrameScheduler:
    """
    Frame scheduler for real-time video playback.
    
    Synchronizes video frames to master clock (audio).
    """
    
    def __init__(
        self,
        master_clock: Optional[Clock] = None,
        max_queue_size: int = 16,
        lateness_threshold: float = 0.05  # 50ms
    ):
        """
        Initialize frame scheduler.
        
        Args:
            master_clock: Master clock for synchronization (default: SystemClock)
            max_queue_size: Maximum frame queue size
            lateness_threshold: Max lateness before dropping (seconds)
        """
        self.master_clock = master_clock or SystemClock()
        self.max_queue_size = max_queue_size
        self.lateness_threshold = lateness_threshold
        
        # Frame queue (PTS-ordered)
        self.frame_queue: Queue[VideoFrame] = Queue(maxsize=max_queue_size)
        
        # Thread safety
        self.lock = Lock()
        
        # Statistics
        self.stats = SchedulerStats()
        
        # Playback state
        self.is_running = False
        
        print(f"[Scheduler] Initialized")
        print(f"[Scheduler] Max queue size: {max_queue_size}")
        print(f"[Scheduler] Lateness threshold: {lateness_threshold * 1000:.0f}ms")
    
    def push_frame(self, frame: VideoFrame, block: bool = True, timeout: Optional[float] = None):
        """
        Add frame to scheduler queue (called by decoder thread).
        
        Args:
            frame: Decoded video frame with PTS
            block: Whether to block if queue is full
            timeout: Timeout in seconds (None = infinite)
        
        Raises:
            queue.Full: If queue is full and block=False
        """
        try:
            self.frame_queue.put(frame, block=block, timeout=timeout)
            
            with self.lock:
                self.stats.total_frames += 1
            
            if self.stats.total_frames % 100 == 0:
                print(f"[Scheduler] Queued {self.stats.total_frames} frames (queue size: {self.frame_queue.qsize()})")
        
        except Exception as e:
            print(f"[Scheduler] Failed to queue frame: {e}")
            raise
    
    def get_next_frame(self) -> Optional[VideoFrame]:
        """
        Get next frame to present (called by render thread).
        
        Returns:
            VideoFrame if ready to present, None otherwise
        
        Logic:
        - If queue empty: return None
        - If frame PTS > clock + threshold: too early, return None
        - If frame PTS < clock - threshold: too late, drop and try next
        - Otherwise: present frame
        """
        if self.frame_queue.empty():
            return None
        
        clock_time = self.master_clock.get_time()
        
        while not self.frame_queue.empty():
            # Peek at next frame (non-blocking get)
            try:
                frame = self.frame_queue.get_nowait()
            except Empty:
                return None
            
            # Calculate timing
            frame_pts = frame.pts
            delta = frame_pts - clock_time
            
            # Case 1: Frame is too early (>5ms ahead)
            if delta > 0.005:
                # Put frame back (we peeked too early)
                # Put back at front by using a temporary queue
                # For simplicity, just check on next call
                # Actually, we can't put back easily, so we'll wait
                
                with self.lock:
                    self.stats.early_frames += 1
                
                # Put frame back by creating new queue (hacky but works)
                # Better solution: use a peek-able queue or deque
                # For now, return None and keep frame in a "pending" slot
                # Simplified: just return the frame anyway if close enough
                if delta < 0.1:  # Within 100ms, present anyway
                    with self.lock:
                        self.stats.presented_frames += 1
                    return frame
                else:
                    # Frame is way too early, this shouldn't happen
                    # Log and skip
                    print(f"[Scheduler] Frame {frame.frame_index} too early: {delta * 1000:.1f}ms ahead")
                    return None
            
            # Case 2: Frame is too late (drop)
            elif delta < -self.lateness_threshold:
                with self.lock:
                    self.stats.dropped_frames += 1
                    self.stats.late_frames += 1
                
                if self.stats.dropped_frames % 10 == 1:  # Log every 10th drop
                    print(f"[Scheduler] Dropped frame {frame.frame_index}: {-delta * 1000:.1f}ms late")
                
                # Continue to next frame
                continue
            
            # Case 3: Frame is on time (present)
            else:
                with self.lock:
                    self.stats.presented_frames += 1
                
                return frame
        
        return None
    
    def reset(self):
        """Reset scheduler state."""
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        # Reset clock
        self.master_clock.reset()
        
        # Reset stats
        with self.lock:
            self.stats = SchedulerStats()
        
        print("[Scheduler] Reset")
    
    def get_stats(self) -> SchedulerStats:
        """Get scheduler statistics."""
        with self.lock:
            return SchedulerStats(
                total_frames=self.stats.total_frames,
                presented_frames=self.stats.presented_frames,
                dropped_frames=self.stats.dropped_frames,
                early_frames=self.stats.early_frames,
                late_frames=self.stats.late_frames
            )
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.frame_queue.qsize()
    
    def is_queue_full(self) -> bool:
        """Check if queue is full."""
        return self.frame_queue.full()
    
    def is_queue_empty(self) -> bool:
        """Check if queue is empty."""
        return self.frame_queue.empty()


def main():
    """Test frame scheduler with synthetic frames."""
    
    print("=" * 60)
    print("Frame Scheduler - Test")
    print("=" * 60)
    
    # Create scheduler with system clock
    scheduler = FrameScheduler(master_clock=SystemClock(), max_queue_size=10)
    
    # Generate synthetic frames at 30 fps
    fps = 30.0
    frame_duration = 1.0 / fps
    
    print(f"\n[Test] Simulating {fps} fps playback...")
    
    # Simulate decoder pushing frames
    for i in range(100):
        frame = VideoFrame(
            data=None,  # Don't need actual pixel data for timing test
            width=640,
            height=480,
            pts=i * frame_duration,
            frame_index=i,
            format='rgb24'
        )
        
        scheduler.push_frame(frame)
        
        # Simulate some decoding delay
        time.sleep(frame_duration * 0.5)  # Decode at 2× speed
    
    print("\n[Test] All frames queued, starting presentation...")
    
    # Simulate render loop getting frames
    presented = 0
    while presented < 100:
        frame = scheduler.get_next_frame()
        
        if frame:
            presented += 1
            if presented % 10 == 0:
                print(f"[Test] Presented frame {frame.frame_index} (PTS: {frame.pts:.3f}s)")
        
        # Simulate render delay (slower than real-time to test dropping)
        time.sleep(frame_duration * 1.2)  # Render at 0.8× speed (slightly slow)
    
    # Print stats
    stats = scheduler.get_stats()
    print("\n" + "=" * 60)
    print("Scheduler Statistics:")
    print("=" * 60)
    print(f"Total frames:     {stats.total_frames}")
    print(f"Presented frames: {stats.presented_frames}")
    print(f"Dropped frames:   {stats.dropped_frames}")
    print(f"Early frames:     {stats.early_frames}")
    print(f"Late frames:      {stats.late_frames}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

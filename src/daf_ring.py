import numpy as np
import math


class DAFRing:
    """Circular buffer for Delayed Auditory Feedback (DAF)"""
    
    def __init__(self, sample_rate=16000, delay_ms=175, buffer_duration_ms=500, bluetooth_adjustment_ms=0):
        """
        Initialize DAF ring buffer
        
        Args:
            sample_rate: Audio sample rate in Hz
            delay_ms: Target total mouth-to-ear DAF delay in milliseconds
            buffer_duration_ms: Total buffer duration in milliseconds
            bluetooth_adjustment_ms: Estimated input+output Bluetooth latency to compensate
        """
        self.sample_rate = sample_rate
        self.delay_ms = delay_ms
        self.bluetooth_adjustment_ms = bluetooth_adjustment_ms
        # Compute ring delay to hit the target mouth-to-ear delay.
        # ring_delay = max(0, target_delay - (estimated_bt_input + estimated_bt_output))
        # Note: This is an estimate; actual stack latencies may vary.
        self.total_delay_ms = max(0, delay_ms - bluetooth_adjustment_ms)
        
        # Calculate buffer parameters
        self.samples_per_frame = int(20 * sample_rate / 1000)  # 20ms frames
        self.buffer_length = int(buffer_duration_ms * sample_rate / 1000)
        self.delay_samples = int(self.total_delay_ms * sample_rate / 1000)
        
        # Initialize circular buffer
        self.buffer = np.zeros(self.buffer_length, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = (self.write_pos - self.delay_samples) % self.buffer_length
        
        # State
        self.active = False
        
        # Audio processing parameters
        self.output_gain = 0.9
        self.limiter_ceiling_db = -3.0
        self.limiter_ceiling_linear = 10 ** (self.limiter_ceiling_db / 20.0)
        
        # Pre-allocate temp arrays for performance
        self.temp_output = np.zeros(self.samples_per_frame, dtype=np.float32)
    
    def push(self, frame_f32):
        """
        Push a frame of audio into the ring buffer
        
        Args:
            frame_f32: Input frame as float32 array of shape (samples_per_frame,)
        """
        if len(frame_f32) != self.samples_per_frame:
            raise ValueError(f"Expected frame length {self.samples_per_frame}, got {len(frame_f32)}")
        
        end = self.write_pos + self.samples_per_frame
        if end <= self.buffer_length:
            self.buffer[self.write_pos:end] = frame_f32
        else:
            first = self.buffer_length - self.write_pos
            self.buffer[self.write_pos:] = frame_f32[:first]
            self.buffer[:self.samples_per_frame - first] = frame_f32[first:]
        self.write_pos = (self.write_pos + self.samples_per_frame) % self.buffer_length
    
    def pull(self):
        """
        Pull a delayed frame from the ring buffer
        
        Returns:
            Delayed frame as float32 array of shape (samples_per_frame,)
        """
        if not self.active:
            return np.zeros(self.samples_per_frame, dtype=np.float32)
        
        end = self.read_pos + self.samples_per_frame
        if end <= self.buffer_length:
            np.copyto(self.temp_output, self.buffer[self.read_pos:end])
        else:
            first = self.buffer_length - self.read_pos
            self.temp_output[:first] = self.buffer[self.read_pos:]
            self.temp_output[first:] = self.buffer[:self.samples_per_frame - first]
        self.read_pos = (self.read_pos + self.samples_per_frame) % self.buffer_length
        
        # Apply hard limiter and output gain
        np.clip(
            self.temp_output,
            -self.limiter_ceiling_linear,
            self.limiter_ceiling_linear,
            out=self.temp_output,
        )
        self.temp_output *= self.output_gain
        
        return self.temp_output.copy()
    
    def set_active(self, active):
        """Set DAF active state"""
        self.active = active
    
    def set_output_gain(self, gain):
        """Set output gain (0.0 to 1.0)"""
        self.output_gain = max(0.0, min(1.0, gain))
    
    def set_limiter_ceiling_db(self, ceiling_db):
        """Set limiter ceiling in dB"""
        self.limiter_ceiling_db = ceiling_db
        self.limiter_ceiling_linear = 10 ** (ceiling_db / 20.0)
    
    def update_bluetooth_adjustment(self, bluetooth_adjustment_ms: int):
        """
        Update Bluetooth latency compensation
        
        Args:
            bluetooth_adjustment_ms: Bluetooth latency to compensate for (subtracted from delay)
        """
        self.bluetooth_adjustment_ms = bluetooth_adjustment_ms
        # Recompute ring delay based on new estimated BT latency
        self.total_delay_ms = max(0, self.delay_ms - bluetooth_adjustment_ms)
        self.delay_samples = int(self.total_delay_ms * self.sample_rate / 1000)
        
        # Recalculate read position
        self.read_pos = (self.write_pos - self.delay_samples) % self.buffer_length

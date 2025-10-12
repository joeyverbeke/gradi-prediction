import numpy as np
import vosk
import os
import threading
from typing import Optional


class ASRVosk:
    """Streaming ASR using Vosk"""
    
    def __init__(self, model_path="models/asr/vosk-model-small-en-us", sample_rate=16000):
        """
        Initialize Vosk ASR
        
        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}")
        
        # Load Vosk model
        try:
            self.model = vosk.Model(model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, sample_rate)
            print(f"Loaded Vosk model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Vosk model: {e}")
        
        # State tracking
        self.rolling_text = ""
        self.last_partial = ""
        self.partial_count = 0
        self._last_rolling_update = ""

        # Thread-safety: Vosk recognizer is not thread-safe; guard all access
        self._lock = threading.Lock()
    
    def accept(self, pcm_int16_bytes):
        """
        Feed PCM audio data to Vosk
        
        Args:
            pcm_int16_bytes: PCM audio data as bytes (int16)
        """
        with self._lock:
            if self.recognizer.AcceptWaveform(pcm_int16_bytes):
                # Consume final result to keep decoder stable (we ignore content)
                _ = self.recognizer.Result()
    
    def get_partial_if_new(self) -> Optional[str]:
        """
        Get new partial result if available
        
        Returns:
            New partial text or None if no new partial
        """
        with self._lock:
            partial = self.recognizer.PartialResult()
        
        # Check if partial has changed (any change, not just length increase)
        if partial != self.last_partial:
            self.last_partial = partial
            self.partial_count += 1
            
            # Extract text from JSON partial result
            partial_text = self._extract_text_from_partial(partial)
            
            # Only update rolling text if partial text is not empty
            if partial_text.strip():
                self._update_rolling_text(partial_text)
            
            return partial_text
        
        return None
    
    def _update_rolling_text(self, partial):
        """
        Update rolling text, keeping last 80 words
        
        Args:
            partial: Current partial text
        """
        # Only update if partial is not empty and different from last update
        if partial.strip() and partial != getattr(self, '_last_rolling_update', ''):
            if self.rolling_text:
                self.rolling_text += " " + partial
            else:
                self.rolling_text = partial
            
            # Store the last update to avoid duplicates
            self._last_rolling_update = partial
            
            # Truncate to last 80 words
            words = self.rolling_text.split()
            if len(words) > 80:
                self.rolling_text = " ".join(words[-80:])
    
    def get_rolling_text(self) -> str:
        """Get current rolling text"""
        return self.rolling_text
    
    def get_current_partial(self) -> str:
        """Get current partial text (may not be new)"""
        with self._lock:
            partial = self.recognizer.PartialResult()
        return self._extract_text_from_partial(partial)
    
    def _extract_text_from_partial(self, partial_json: str) -> str:
        """
        Extract text from Vosk partial JSON result
        
        Args:
            partial_json: JSON string from Vosk PartialResult()
            
        Returns:
            Extracted text string
        """
        try:
            import json
            # Parse JSON
            data = json.loads(partial_json)
            
            # Check for "partial" field first (most common)
            if "partial" in data:
                return data["partial"]
            
            # Check for "text" field (final results)
            if "text" in data:
                return data["text"]
            
            # Fallback to empty string
            return ""
            
        except (json.JSONDecodeError, KeyError):
            # If JSON parsing fails, try to extract text directly
            # This handles cases where the result might be plain text
            return partial_json.strip()
    
    def reset(self):
        """Reset ASR state"""
        with self._lock:
            self.rolling_text = ""
            self.last_partial = ""
            self.partial_count = 0
            self._last_rolling_update = ""
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
    
    def convert_float32_to_int16_bytes(self, frame_f32):
        """
        Convert float32 frame to int16 PCM bytes for Vosk
        
        Args:
            frame_f32: Float32 audio frame
            
        Returns:
            Int16 PCM bytes
        """
        # Clip to [-1, 1] and convert to int16
        frame_clipped = np.clip(frame_f32, -1.0, 1.0)
        frame_int16 = (frame_clipped * 32767.0).astype(np.int16)
        
        return frame_int16.tobytes()

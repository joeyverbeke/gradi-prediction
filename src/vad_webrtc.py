"""Thin wrapper around the webrtcvad module with simple hysteresis."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    webrtcvad = None  # type: ignore


@dataclass
class WebRTCVADConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    aggressiveness: int = 2
    activation_frames: int = 3
    release_frames: int = 5
    # --- Adaptive energy gate (Change 6, ported from Mediate's Change 1) ---
    # All inert unless energy_gate_enabled, so the gate-off path is byte-for-byte
    # identical to the pre-gate behavior. Ratios are the multiple of the rolling
    # noise floor a frame's RMS must clear (in addition to WebRTC VAD saying
    # speech): onset_ratio while inactive, offset_ratio (lower, for hysteresis)
    # while active. The floor is an EMA over gate-classified non-speech frames
    # with slow-rise / fast-fall asymmetry, seeded from the first
    # ~floor_init_seconds of audio.
    energy_gate_enabled: bool = False
    onset_ratio: Optional[float] = None
    offset_ratio: Optional[float] = None
    floor_alpha_rise: float = 0.02
    floor_alpha_fall: float = 0.10
    floor_init_seconds: float = 1.0


class WebRTCVAD:
    """Maintains voiced/silent state using WebRTC VAD decisions."""

    def __init__(self, config: WebRTCVADConfig):
        if webrtcvad is None:
            raise ImportError("webrtcvad package is not installed")

        if config.frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD requires 10, 20, or 30ms frames")

        if config.aggressiveness < 0 or config.aggressiveness > 3:
            raise ValueError("VAD aggressiveness must be between 0 and 3")

        self._config = config
        self._vad = webrtcvad.Vad(config.aggressiveness)
        self._required_voiced = max(1, config.activation_frames)
        self._required_silence = max(1, config.release_frames)
        self._voiced_run = 0
        self._silent_run = self._required_silence
        self._active = False
        self._expected_frame_bytes = int(config.sample_rate * config.frame_ms / 1000) * 2

        # --- Adaptive energy gate state (Change 6) ---
        # Inert when disabled: _gate_enabled short-circuits process() to the
        # pre-gate path.
        self._gate_enabled = config.energy_gate_enabled
        self._onset_ratio = config.onset_ratio
        self._offset_ratio = config.offset_ratio
        self._floor_alpha_rise = config.floor_alpha_rise
        self._floor_alpha_fall = config.floor_alpha_fall
        if self._gate_enabled and (self._onset_ratio is None or self._offset_ratio is None):
            raise ValueError("energy gate enabled but onset_ratio/offset_ratio not set")
        frame_duration_s = config.frame_ms / 1000.0
        self._init_frames_needed = (
            max(1, round(config.floor_init_seconds / frame_duration_s))
            if self._gate_enabled
            else 0
        )
        self._reset_floor()

    def _reset_floor(self) -> None:
        """Blank the noise-floor estimate so it re-seeds from the next ~1 s of audio."""
        self._floor_rms: Optional[float] = None
        self._floor_ready = False
        self._init_rms_sum = 0.0
        self._init_rms_count = 0

    @property
    def floor_rms(self) -> Optional[float]:
        """Current noise-floor estimate, or None when the gate is off / not yet seeded."""
        if self._gate_enabled and self._floor_ready:
            return self._floor_rms
        return None

    def _frame_rms(self, frame_bytes: bytes) -> float:
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float64)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples * samples)))

    def process(self, frame_bytes: bytes) -> bool:
        """Update state with a PCM16 frame and return aggregated speech flag."""
        if len(frame_bytes) != self._expected_frame_bytes:
            raise ValueError(
                f"Expected frame of {self._expected_frame_bytes} bytes, got {len(frame_bytes)}"
            )

        vad_speech = self._vad.is_speech(frame_bytes, self._config.sample_rate)

        if not self._gate_enabled:
            # Gate off: identical to the pre-gate behavior.
            gated = vad_speech
        else:
            rms = self._frame_rms(frame_bytes)
            if not self._floor_ready:
                # Blind init window (~floor_init_seconds): seed the floor and
                # suppress activation until the gate arms.
                self._init_rms_sum += rms
                self._init_rms_count += 1
                if self._init_rms_count >= self._init_frames_needed:
                    self._floor_rms = self._init_rms_sum / self._init_rms_count
                    self._floor_ready = True
                gated = False
            else:
                ratio = self._offset_ratio if self._active else self._onset_ratio
                threshold = self._floor_rms * ratio
                gated = vad_speech and rms >= threshold
                # Update the floor ONLY from gate-classified non-speech frames,
                # and NEVER while active (the core invariant).
                if not self._active and not gated:
                    alpha = (
                        self._floor_alpha_rise
                        if rms > self._floor_rms
                        else self._floor_alpha_fall
                    )
                    self._floor_rms = (1.0 - alpha) * self._floor_rms + alpha * rms

        if gated:
            self._voiced_run = min(self._required_voiced, self._voiced_run + 1)
            self._silent_run = 0
        else:
            self._silent_run = min(self._required_silence, self._silent_run + 1)
            self._voiced_run = 0

        if not self._active and self._voiced_run >= self._required_voiced:
            self._active = True
        elif self._active and self._silent_run >= self._required_silence:
            self._active = False

        return self._active

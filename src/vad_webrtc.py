"""Thin wrapper around the webrtcvad module with simple hysteresis."""

from dataclasses import dataclass

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

    def process(self, frame_bytes: bytes) -> bool:
        """Update state with a PCM16 frame and return aggregated speech flag."""
        if len(frame_bytes) != self._expected_frame_bytes:
            raise ValueError(
                f"Expected frame of {self._expected_frame_bytes} bytes, got {len(frame_bytes)}"
            )

        is_voiced = self._vad.is_speech(frame_bytes, self._config.sample_rate)

        if is_voiced:
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


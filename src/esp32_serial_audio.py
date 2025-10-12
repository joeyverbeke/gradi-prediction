"""Serial audio ingestion for ESP32 DAF firmware."""

import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    serial = None  # type: ignore
    SERIAL_IMPORT_ERROR = exc
else:
    SERIAL_IMPORT_ERROR = None


class ESP32SerialAudio:
    """Read PCM chunks from ESP32 over serial and emit 20ms frames."""

    def __init__(
        self,
        serial_port: str,
        baud_rate: int = 921_600,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        chunk_samples: int = 1024,
        input_gain: float = 1.0,
        vad_enabled: bool = True,
        vad_rms_threshold: float = 0.015,
        log_callback: Optional[Callable[[str], None]] = None,
        state_callback: Optional[Callable[[bool], None]] = None,
    ) -> None:
        if SERIAL_IMPORT_ERROR is not None:
            raise ImportError(
                "pyserial is required for ESP32SerialAudio"  # pragma: no cover - init guard
            ) from SERIAL_IMPORT_ERROR

        self.serial_port_path = serial_port
        self.baud_rate = baud_rate
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.chunk_samples = chunk_samples
        self.input_gain = input_gain
        self.vad_enabled = vad_enabled
        self.vad_rms_threshold = vad_rms_threshold
        self.log_callback = log_callback or (lambda msg: print(f"ESP32SerialAudio: {msg}"))
        self.state_callback = state_callback

        self.samples_per_frame = int(frame_ms * sample_rate / 1000)
        if self.samples_per_frame <= 0:
            raise ValueError("frame_ms must be large enough to contain at least one sample")

        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=200)
        self._serial: Optional[serial.Serial] = None  # type: ignore[assignment]
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        self._buffer_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending_blocks: "deque[np.ndarray]" = deque()
        self._pending_available = 0
        self._frame_builder = np.empty(self.samples_per_frame, dtype=np.float32)
        self._last_voice_time_ms = 0.0
        self._current_daf_state = False

    # Public API ---------------------------------------------------------
    def start(self) -> None:
        """Open serial connection and start reader thread."""
        if self._running:
            return

        self._serial = serial.Serial(  # type: ignore[call-arg]
            self.serial_port_path,
            baudrate=self.baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self.log_callback(
            f"Connected to {self.serial_port_path} @ {self.baud_rate} baud (chunk {self.chunk_samples} samples)"
        )

    def stop(self) -> None:
        """Stop reader thread and close serial port."""
        self._running = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        self._reader_thread = None

        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self.log_callback("Serial connection closed")

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        with self._buffer_lock:
            self._pending_blocks.clear()
            self._pending_available = 0

    def get_queue_size(self) -> int:
        return self.frame_queue.qsize()

    def get_last_voice_time_ms(self) -> float:
        return self._last_voice_time_ms

    def set_daf_ring(self, _ring) -> None:
        """Compat shim for previous AudioIO API."""
        # ESP32 handles playback directly; ring unused.
        return

    def get_current_daf_state(self) -> bool:
        return self._current_daf_state

    def set_state_callback(self, callback: Optional[Callable[[bool], None]]) -> None:
        self.state_callback = callback

    def request_state_sync(self) -> None:
        self.send_command("STATE?")

    def enable_daf(self) -> None:
        self._send_daf_command(True)

    def disable_daf(self) -> None:
        self._send_daf_command(False)

    def _send_daf_command(self, enable: bool) -> None:
        command = f"DAF {'ON' if enable else 'OFF'}"
        self.send_command(command)

    def send_command(self, command: str) -> None:
        """Send a line-oriented command to ESP32."""
        payload = f"{command.strip()}\n".encode("ascii", errors="ignore")

        if not self._serial or not self._serial.is_open:
            raise RuntimeError("Serial port not open")

        with self._write_lock:
            try:
                self._serial.write(payload)
                self._serial.flush()
            except Exception as exc:
                self.log_callback(f"Failed to send '{command}': {exc}")
                raise

    # Internal helpers ---------------------------------------------------
    def _reader_loop(self) -> None:
        assert self._serial is not None
        ser = self._serial
        while self._running:
            try:
                header_bytes = ser.readline()
                if not header_bytes:
                    continue

                try:
                    header = header_bytes.decode("ascii", errors="ignore").strip()
                except UnicodeDecodeError:
                    continue

                if not header:
                    continue

                if header.startswith("AUDIO "):
                    parts = header.split()
                    if len(parts) < 2:
                        continue
                    try:
                        payload_bytes = int(parts[1])
                    except ValueError:
                        continue
                    self._read_audio_payload(ser, payload_bytes)
                elif header.startswith("STATE "):
                    state = header.split(" ", 1)[1].upper() == "ON"
                    self._notify_state(state)
                elif header.startswith("LOG "):
                    self.log_callback(header[4:])
                else:
                    # Unexpected line, surface for debugging
                    self.log_callback(f"Unrecognized header: {header}")
            except Exception as exc:
                if self._running:
                    self.log_callback(f"Serial read error: {exc}")
                    time.sleep(0.1)

    def _read_audio_payload(self, ser: serial.Serial, payload_bytes: int) -> None:  # type: ignore[name-defined]
        remaining = payload_bytes
        chunks: list[bytes] = []
        while remaining > 0 and self._running:
            chunk = ser.read(remaining)
            if not chunk:
                # Give the device time to push remaining data
                time.sleep(0.001)
                continue
            chunks.append(chunk)
            remaining -= len(chunk)

        if remaining != 0:
            self.log_callback(
                f"Expected {payload_bytes} audio bytes but stream ended early"
            )
            return

        pcm_bytes = b"".join(chunks)
        self._handle_pcm_chunk(pcm_bytes)

    def _handle_pcm_chunk(self, pcm_bytes: bytes) -> None:
        if len(pcm_bytes) % 2 != 0:
            self.log_callback("Dropping odd-length PCM payload")
            return

        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        if pcm.size == 0:
            return

        samples = pcm.astype(np.float32) / 32768.0
        if self.input_gain != 1.0:
            samples *= self.input_gain

        with self._buffer_lock:
            if samples.size:
                self._pending_blocks.append(samples)
                self._pending_available += samples.size

            while self._pending_available >= self.samples_per_frame:
                needed = self.samples_per_frame
                filled = 0
                while filled < needed:
                    block = self._pending_blocks[0]
                    take = min(block.shape[0], needed - filled)
                    self._frame_builder[filled : filled + take] = block[:take]
                    if take == block.shape[0]:
                        self._pending_blocks.popleft()
                    else:
                        self._pending_blocks[0] = block[take:]
                    filled += take
                    self._pending_available -= take

                self._emit_frame(self._frame_builder.copy())

    def _emit_frame(self, frame: np.ndarray) -> None:
        if frame.size != self.samples_per_frame:
            return

        if self.vad_enabled:
            rms = float(np.sqrt(np.mean(frame * frame)))
            if rms >= self.vad_rms_threshold:
                self._last_voice_time_ms = time.time() * 1000.0

        try:
            if not frame.flags.owndata:
                frame = frame.copy()
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Drop frame when consumer is lagging to keep real-time behavior
            pass

    def _notify_state(self, state: bool) -> None:
        self._current_daf_state = state
        if self.state_callback:
            try:
                self.state_callback(state)
            except Exception as exc:
                self.log_callback(f"State callback error: {exc}")

    # Context manager helpers --------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

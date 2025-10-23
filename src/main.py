#!/usr/bin/env python3
"""
MVP Speech Monitoring Prototype for macOS
Main orchestrator that coordinates audio, ASR, LLM prediction, and DAF triggering.
"""

import argparse
import yaml
import time
import threading
import queue
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

# Local imports
from daf_ring import DAFRing
from audio_io import AudioIO
from asr_vosk import ASRVosk
from predictor_llamacpp import PredictorLlamaCPP
from detector_keywords import KeywordDetector
from state import DAFState
from device_detector import DeviceDetector
import utils

try:
    from esp32_serial_audio import ESP32SerialAudio
except ImportError:
    ESP32SerialAudio = None  # Optional dependency for ESP32 integration


class SpeechMonitor:
    """Main speech monitoring system"""

    def __init__(self, config_dir="config", log_level: str = "ERROR", force_cpu: bool = False):
        """Initialize speech monitoring system"""
        self.config_dir = config_dir
        self.log_level = log_level.upper()
        self.force_cpu = force_cpu
        
        # Load configurations
        self.audio_config = self._load_config("audio.yml")
        self.keywords_config = self._load_config("keywords.yml")
        
        # Initialize components
        self.daf_ring = None
        self.audio_io = None
        self.asr = None
        self.predictor = None
        self.detector = None
        self.state = None

        # ASR worker thread
        self.asr_thread = None
        self.asr_running = False
        self.asr_queue = queue.Queue(maxsize=200)
        
        # State tracking
        self.partial_text = ""
        self.horizon_text = ""
        self.last_partial_length = 0
        self.trigger_ring = deque(maxlen=6)  # Last 6 trigger results
        self.last_llm_call_time = 0
        self.llm_call_interval_ms = 100
        self._last_hit_state = (False, False)
        self._last_hit_log_ms = 0
        self._last_horizon_logged = ""
        self._llm_executor: Optional[ThreadPoolExecutor] = None
        self._llm_future: Optional[Future] = None
        self._llm_lock = threading.Lock()
        
        # ASR monitoring
        self.last_partial_update_time = 0
        self.asr_stuck_threshold_ms = 5000  # 5 seconds without updates
        
        # Performance tracking
        self.start_time = None
        self.frame_count = 0

        # Debug mode
        self.debug_mode = True  # Set to True for verbose logging

        # ESP32 integration state
        self.using_esp_audio = False
        self.remote_daf_state = False
        self._remote_state_lock = threading.Lock()
        
    def _load_config(self, filename: str) -> dict:
        """Load YAML configuration file"""
        config_path = f"{self.config_dir}/{filename}"
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load config {config_path}: {e}")
    
    def initialize(self):
        """Initialize all system components"""
        utils.setup_logging(self.log_level)
        utils.log_audio_status("Initializing system...")
        
        try:
            audio_source = self.audio_config.get('audio_source', 'local').lower()

            if audio_source == 'esp32_serial':
                if ESP32SerialAudio is None:
                    raise RuntimeError(
                        "ESP32 serial audio requested but pyserial is not installed"
                    )

                utils.log_audio_status("ESP32 serial audio mode selected")
                self.using_esp_audio = True

                # Device detector is not required when audio originates on ESP32
                self.device_detector = None
                bluetooth_adjustment = 0

                serial_path = self.audio_config.get('esp_serial_port')
                if not serial_path:
                    raise RuntimeError(
                        "audio.yml missing esp_serial_port for ESP32 serial audio mode"
                    )

                # Initialize DAF ring for trigger/hold tracking only
                self.daf_ring = DAFRing(
                    sample_rate=self.audio_config['sample_rate'],
                    delay_ms=self.audio_config['daf_delay_ms'],
                    bluetooth_adjustment_ms=0
                )
                self.daf_ring.set_output_gain(self.audio_config['output_gain'])
                self.daf_ring.set_limiter_ceiling_db(self.audio_config['limiter_ceiling_db'])

                self.audio_io = ESP32SerialAudio(
                    serial_port=serial_path,
                    baud_rate=self.audio_config.get('esp_serial_baud', 921600),
                    sample_rate=self.audio_config['sample_rate'],
                    frame_ms=self.audio_config['frame_ms'],
                    chunk_samples=self.audio_config.get('esp_chunk_samples', 1024),
                    input_gain=self.audio_config['input_gain'],
                    vad_enabled=self.audio_config.get('vad_enabled', True),
                    vad_rms_threshold=self.audio_config.get('vad_rms_threshold', 0.015),
                    log_callback=utils.log_audio_status,
                    state_callback=self._handle_esp_state_update,
                )
                self.audio_io.set_daf_ring(self.daf_ring)
            else:
                # Initialize device detector
                self.device_detector = DeviceDetector()

                # Get Bluetooth adjustment
                bluetooth_adjustment = self.device_detector.get_daf_adjustment(
                    self.audio_config.get('input_device'),
                    self.audio_config.get('output_device')
                )

                # Log device information
                device_info = self.device_detector.get_current_devices_info()
                utils.log_audio_status(f"Input device: {device_info['input']['name']} (Bluetooth: {device_info['input']['is_bluetooth']})")
                utils.log_audio_status(f"Output device: {device_info['output']['name']} (Bluetooth: {device_info['output']['is_bluetooth']})")
                utils.log_audio_status(f"DAF compensation: -{bluetooth_adjustment}ms for Bluetooth latency")

                # Initialize DAF ring buffer
                self.daf_ring = DAFRing(
                    sample_rate=self.audio_config['sample_rate'],
                    delay_ms=self.audio_config['daf_delay_ms'],
                    bluetooth_adjustment_ms=bluetooth_adjustment
                )
                self.daf_ring.set_output_gain(self.audio_config['output_gain'])
                self.daf_ring.set_limiter_ceiling_db(self.audio_config['limiter_ceiling_db'])
                utils.log_audio_status(
                    f"DAF target: {self.audio_config['daf_delay_ms']}ms, BT est: {bluetooth_adjustment}ms, ring delay applied: {self.daf_ring.total_delay_ms}ms"
                )

                # Initialize audio I/O
                self.audio_io = AudioIO(
                    sample_rate=self.audio_config['sample_rate'],
                    frame_ms=self.audio_config['frame_ms'],
                    input_gain=self.audio_config['input_gain'],
                    input_device=self.audio_config['input_device'],
                    output_device=self.audio_config['output_device'],
                    vad_enabled=self.audio_config.get('vad_enabled', True),
                    vad_rms_threshold=self.audio_config.get('vad_rms_threshold', 0.015)
                )
                self.audio_io.set_daf_ring(self.daf_ring)

            # Initialize ASR
            self.asr = ASRVosk(
                sample_rate=self.audio_config['sample_rate']
            )
            
            # Initialize LLM predictor
            n_gpu_layers = self.keywords_config.get('llm_gpu_layers')
            if self.force_cpu:
                utils.log_audio_status("CPU-only mode requested; llama.cpp will run on CPU")
                n_gpu_layers = 0

            self.predictor = PredictorLlamaCPP(
                context_tokens=self.keywords_config['context_tokens'],
                n_gpu_layers=n_gpu_layers
            )
            
            # Initialize keyword detector
            self.detector = KeywordDetector.from_config(
                self.keywords_config['stems']
            )
            
            # Initialize DAF state
            self.state = DAFState(
                hold_ms=self.audio_config['hold_ms']
            )
            
            utils.log_audio_status("All components initialized successfully")
            if not self._llm_executor:
                self._llm_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
            
        except Exception as e:
            utils.log_error("Initialization failed", e)
            raise
    
    def start_asr_worker(self):
        """Start ASR worker thread"""
        if self.asr_thread and self.asr_thread.is_alive():
            return
        
        self.asr_running = True
        self.asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
        self.asr_thread.start()
        utils.log_audio_status("ASR worker thread started")
    
    def _asr_worker(self):
        """ASR worker thread function"""
        frame_count = 0
        while self.asr_running:
            try:
                # Get frame from audio I/O
                frame = self.audio_io.get_frame(timeout=0.1)
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Convert to PCM bytes for Vosk
                pcm_bytes = self.asr.convert_float32_to_int16_bytes(frame)
                
                # Feed to ASR
                self.asr.accept(pcm_bytes)
                
                # Check for new partial
                new_partial = self.asr.get_partial_if_new()
                if new_partial:
                    self.partial_text = new_partial
                    utils.log_partial(new_partial)
                
                # Log frame processing every 1000 frames (about 20 seconds)
                if frame_count % 1000 == 0:
                    queue_size = self.audio_io.get_queue_size()
                    utils.log_audio_status(f"ASR worker processed {frame_count} frames, queue size: {queue_size}")
                
            except Exception as e:
                utils.log_error("ASR worker error", e)
                time.sleep(0.01)
    
    def _should_call_llm(self) -> bool:
        """Determine if LLM should be called based on partial growth"""
        current_time = utils.get_timestamp_ms()
        
        # Check time interval
        if current_time - self.last_llm_call_time < self.llm_call_interval_ms:
            return False
        
        # Check if partial has grown significantly
        if len(self.partial_text) - self.last_partial_length >= 4:
            self.last_llm_call_time = current_time
            self.last_partial_length = len(self.partial_text)
            return True
        
        return False
    
    def _get_partial_tail(self, words: int = 15) -> str:
        """Get last N words from partial text"""
        if not self.partial_text:
            return ""
        # When DAF is active and we latch until silence, don't truncate the partial
        if self.state and self.state.is_active() and self.audio_config.get('daf_latch_until_silence', True):
            return self.partial_text
        words_list = self.partial_text.split()
        if len(words_list) <= words:
            return self.partial_text
        return " ".join(words_list[-words:])
    
    def _update_trigger_ring(self, hit: bool):
        """Update trigger ring buffer"""
        self.trigger_ring.append(hit)
    
    def _check_trigger_condition(self) -> bool:
        """Check if trigger condition is met"""
        if len(self.trigger_ring) < self.audio_config['consecutive_hits']:
            return False
        
        # Check for consecutive hits
        recent_hits = list(self.trigger_ring)[-self.audio_config['consecutive_hits']:]
        return all(recent_hits)
    
    def update_cycle(self):
        """Main update cycle - called every 40ms"""
        try:
            current_time = utils.get_timestamp_ms()

            # Skip heavy processing when we're clearly idle (no recent partials or voice activity).
            last_voice_ms = self.audio_io.get_last_voice_time_ms() if self.audio_io else 0
            recent_voice = last_voice_ms and (current_time - last_voice_ms) < 200

            # Get current partial text (always available)
            current_partial = self.asr.get_current_partial()
            if current_partial != self.partial_text:
                self.partial_text = current_partial
                self.last_partial_update_time = current_time
                # Log when partial text changes (only non-empty for signal)
                if current_partial.strip():
                    utils.log_partial(current_partial)
                else:
                    # Clear stale horizon when partial becomes empty
                    self.horizon_text = ""
                    self._last_horizon_logged = ""
            skip_heavy = (
                not recent_voice
                and not self.partial_text.strip()
                and (not self.state or not self.state.is_active())
            )
            
            # Debug logging
            if self.debug_mode and self.frame_count % 100 == 0:  # Every 4 seconds
                utils.log_audio_status(f"Debug - Partial: '{self.partial_text}', Rolling: '{self.asr.get_rolling_text()[:30]}...'")
            
            # Check if ASR is stuck (no updates for a while)
            if (self.last_partial_update_time > 0 and 
                current_time - self.last_partial_update_time > self.asr_stuck_threshold_ms):
                utils.log_error("ASR appears stuck, resetting...")
                self.asr.reset()
                self.last_partial_update_time = current_time

            if skip_heavy:
                return
            
            # Harvest completed LLM predictions
            if self._llm_future and self._llm_future.done():
                try:
                    result = self._llm_future.result()
                    if result:
                        if result != self._last_horizon_logged:
                            utils.log_horizon(result)
                            self._last_horizon_logged = result
                        self.horizon_text = result
                    else:
                        self.horizon_text = ""
                        self._last_horizon_logged = ""
                except Exception as future_exc:
                    utils.log_error("LLM prediction error", future_exc)
                finally:
                    self._llm_future = None
            
            # Check if LLM should be called
            if self._should_call_llm() and self._llm_executor:
                # Get rolling text for context
                rolling_text = self.asr.get_rolling_text()
                if rolling_text and not self._llm_future:
                    # Call LLM predictor
                    params = {
                        'min_pred_tokens': self.keywords_config['min_pred_tokens'],
                        'max_pred_tokens': self.keywords_config['max_pred_tokens'],
                        'top_k': self.keywords_config['top_k']
                    }
                    
                    self._llm_future = self._llm_executor.submit(
                        self.predictor.predict_horizon,
                        rolling_text,
                        params,
                    )
                elif not rolling_text:
                    self.horizon_text = ""
                    self._last_horizon_logged = ""
            
            # Get partial tail for scanning
            partial_tail = self._get_partial_tail()
            
            # Scan for keywords
            hit_asr = self.detector.scan(partial_tail)
            hit_llm = self.detector.scan(self.horizon_text) if self.horizon_text else False
            
            # Log hit detection
            hit_state = (hit_asr, hit_llm)
            if hit_asr or hit_llm or hit_state != self._last_hit_state:
                utils.log_hit_detection(hit_asr, hit_llm, partial_tail, self.horizon_text)
                self._last_hit_log_ms = current_time
            self._last_hit_state = hit_state
            
            # Update trigger ring
            hit = hit_asr or hit_llm
            self._update_trigger_ring(hit)
            
            # Check trigger condition
            if self._check_trigger_condition() and not self.state.is_active():
                self.state.on()
                self._apply_daf_transition(True, "Trigger condition met")

            # VAD-based deactivation: if silence for configured window, close DAF early
            if self.state.is_active() and self.audio_config.get('vad_enabled', True):
                last_voice_ms = self.audio_io.get_last_voice_time_ms()
                silence_ms = current_time - last_voice_ms if last_voice_ms > 0 else float('inf')
                silence_window_ms = self.audio_config.get('vad_silence_ms', 1000)
                if silence_ms >= silence_window_ms:
                    self.state.off()
                    self._apply_daf_transition(False, f"VAD silence ({int(silence_ms)}ms >= {silence_window_ms}ms)")
                    # Reset trigger ring to avoid immediate re-trigger from stale hits
                    try:
                        self.trigger_ring.clear()
                    except Exception:
                        self.trigger_ring = deque(maxlen=6)
                    # Clear horizon text to avoid stale LLM hits
                    self.horizon_text = ""
                    self._last_horizon_logged = ""

            # Update DAF state (hold timer) unless latching until silence
            latch_until_silence = self.audio_config.get('daf_latch_until_silence', True) and self.audio_config.get('vad_enabled', True)
            if not latch_until_silence:
                if not self.state.update():
                    self._apply_daf_transition(False, "Hold timer expired")

        except Exception as e:
            utils.log_error("Update cycle error", e)

    def run(self):
        """Run the speech monitoring system"""
        utils.log_audio_status("Starting speech monitoring")
        
        try:
            # Start audio I/O
            self.audio_io.start()

            if self.using_esp_audio:
                try:
                    self.audio_io.request_state_sync()
                except Exception as e:
                    utils.log_error("Failed to sync ESP32 state", e)

            # Start ASR worker
            self.start_asr_worker()
            
            # Main loop
            self.start_time = utils.get_timestamp_ms()
            cycle_interval_ms = 40  # 40ms update cycle
            last_cycle_time = self.start_time
            
            while True:
                current_time = utils.get_timestamp_ms()
                # Run update cycle every 40ms
                if current_time - last_cycle_time >= cycle_interval_ms:
                    self.update_cycle()
                    last_cycle_time = current_time
                    self.frame_count += 1
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
            
        except KeyboardInterrupt:
            utils.log_audio_status("Interrupted by user")
        except Exception as e:
            utils.log_error("Runtime error", e)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up system resources"""
        utils.log_audio_status("Cleaning up...")
        if self._llm_executor:
            self._llm_executor.shutdown(wait=False)
            self._llm_executor = None
            self._llm_future = None
        
        # Stop ASR worker
        self.asr_running = False
        if self.asr_thread and self.asr_thread.is_alive():
            self.asr_thread.join(timeout=1.0)
        
        # Stop audio I/O
        if self.audio_io:
            if self.using_esp_audio:
                with self._remote_state_lock:
                    remote_active = self.remote_daf_state
                if remote_active:
                    try:
                        self.audio_io.disable_daf()
                    except Exception as e:
                        utils.log_error("Failed to disable ESP32 DAF during cleanup", e)
            self.audio_io.stop()
        
        # Log final statistics
        if self.start_time:
            total_time = utils.get_timestamp_ms() - self.start_time
        utils.log_audio_status(f"Run completed: {utils.format_duration_ms(total_time)}, {self.frame_count} cycles")

    def _apply_daf_transition(self, active: bool, reason: str) -> None:
        """Toggle DAF either locally or on ESP32."""
        state_changed = False
        if self.daf_ring and self.daf_ring.active != active:
            state_changed = True
            self.daf_ring.set_active(active)

        if self.using_esp_audio and self.audio_io:
            try:
                with self._remote_state_lock:
                    remote_state = self.remote_daf_state
                if active and not remote_state:
                    self.audio_io.enable_daf()
                    with self._remote_state_lock:
                        self.remote_daf_state = True
                    state_changed = True
                elif not active and remote_state:
                    self.audio_io.disable_daf()
                    with self._remote_state_lock:
                        self.remote_daf_state = False
                    state_changed = True
            except Exception as exc:
                utils.log_error("Failed to command ESP32 DAF state", exc)

        if state_changed:
            utils.log_daf_transition(active, reason)

    def _handle_esp_state_update(self, active: bool) -> None:
        with self._remote_state_lock:
            previous = self.remote_daf_state
            self.remote_daf_state = active
        if previous != active:
            utils.log_audio_status(f"ESP32 reported DAF {'ON' if active else 'OFF'}")


def main():
    """Main entry point"""
    print("MVP Speech Monitoring Prototype for macOS")
    print("==========================================")
    
    parser = argparse.ArgumentParser(description="Gradi speech monitoring prototype")
    parser.add_argument(
        "--logging",
        action="store_true",
        help="Enable detailed INFO-level logging (defaults to warnings only)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force llama.cpp horizon predictor to run on CPU only",
    )
    args = parser.parse_args()
    
    # Check for required model files
    required_files = [
        "models/asr/vosk-model-small-en-us",
        "models/llm/llama-3.2-1b-q4_k_m.gguf"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("ERROR: Missing required model files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease download and place the required model files in the models/ directory.")
        return 1
    
    # Create and run speech monitor
    log_level = "INFO" if args.logging else "ERROR"
    monitor = SpeechMonitor(log_level=log_level, force_cpu=args.cpu_only)
    
    try:
        monitor.initialize()
        monitor.run()
        return 0
    except Exception as e:
        utils.log_error("Fatal error", e)
        return 1


if __name__ == "__main__":
    import os
    exit(main())

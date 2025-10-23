# Gradi Prediction Speech Monitor

An integrated desktop + ESP32 system that continuously captures voice from the ESP32-S3 `esp-daf` firmware, transcribes it on the host, predicts risky phrases, and commands on-device delayed auditory feedback (DAF) with a 175 ms delay.

## Directory Layout

```
gradi-prediction/
├── config/                # YAML configuration (audio + keywords)
├── firmware/
│   └── esp32/esp-daf/     # Arduino sketch for the XIAO ESP32S3 DAF firmware
├── models/                # Place ASR + LLM model assets here
└── src/                   # Python sources (main orchestrator + modules)
```

`models/` currently mirrors the working setup in this repository. Replace the contents as needed with your preferred Vosk + LLaMA checkpoints.

## Host Environment Setup

The project targets Python 3.11 and has been verified on Ubuntu 24.04 (WSL2).

```bash
cd gradi-prediction
uv venv --python 3.11
source .venv/bin/activate
uv pip install numpy==1.26.4 sounddevice==0.4.7 soundfile==0.12.1 vosk==0.3.44 \
    pyahocorasick==2.1.0 llama-cpp-python==0.2.90 onnxruntime==1.18.1 \
    pyyaml==6.0.2 loguru==0.7.2 pyserial==3.5 webrtcvad==2.0.10
```

If you plan to use the local microphone fallback (`audio_source: local`), install PortAudio system libraries first:

```bash
sudo apt install libportaudio2
```

### Optional: GPU Acceleration for llama.cpp

With the bundled config the desktop app will auto-offload llama.cpp to GPU whenever the CUDA-enabled wheel is installed (`llm_gpu_layers: -1`). Use the `--cpu-only` flag at launch if you need to keep inference on the CPU for a session.

If you want to rebuild `llama-cpp-python` with CUDA enabled, keep `llm_gpu_layers` at `-1` for full auto offload (or set it to `0` when building a CPU-only wheel).

```bash
source .venv/bin/activate
sudo apt install build-essential cmake ninja-build

export LLAMA_CUDA_ARCH_LIST="120"              # replace with your card's compute capability ×10
export CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on"
export FORCE_CMAKE=1

uv pip install --force-reinstall --no-cache-dir \
    --no-binary llama-cpp-python llama-cpp-python==0.2.90
```

Verify the build:

```bash
PYTHONPATH=src python - <<'PY'
from predictor_llamacpp import PredictorLlamaCPP
_ = PredictorLlamaCPP(n_gpu_layers=-1)
PY
```

## ESP32 Firmware

1. Open `firmware/esp32/esp-daf/esp-daf.ino` in Arduino IDE (or PlatformIO).
2. Select the Seeed XIAO ESP32S3 (or your target ESP32-S3 board) and set the serial monitor baud to `921600`.
3. Flash the sketch. On reset the firmware will log to serial and wait for commands from the desktop.

## Models

The desktop application expects:

- English ASR: `models/asr/vosk-model-small-en-us/`
- Korean ASR: `models/asr/vosk-model-small-ko-0.22/`
- English horizon LLM: `models/llm/llama-3.2-1b-q4_k_m.gguf`
- Korean horizon LLM: `models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf`

Adjust paths in `src/main.py` if you host the assets elsewhere.

## Configuration

Key settings live in `config/audio.yml` (audio pipeline + ESP32 serial) and the language-specific keyword files (`config/keywords_en.yml`, `config/keywords_ko.yml`).

`config/audio.yml` (defaults shown):

```yaml
audio_source: esp32_serial
sample_rate: 16000
frame_ms: 20
daf_delay_ms: 175
hold_ms: 600
consecutive_hits: 2
input_gain: 1.0
output_gain: 0.9
limiter_ceiling_db: -3.0
vad_enabled: true
vad_rms_threshold: 0.015
vad_silence_ms: 1000
daf_latch_until_silence: true
speech_release_ms: 750
partial_activity_ms: 800
daf_max_active_ms: 5000
daf_activation_squelch_ms: 300
webrtc_vad:
  enabled: true
  aggressiveness: 2
  activation_frames: 3
  release_frames: 5
esp_serial_port: /dev/ttyACM0
esp_serial_baud: 921600
esp_chunk_samples: 1024
```

Update `esp_serial_port` to match the device exposed when the ESP32 is connected (`ls /dev/ttyACM*` on Ubuntu). If you switch back to host audio capture, change `audio_source` to `local` and configure `input_device`/`output_device`.

Pick the keyword file that matches your locale (`config/keywords_en.yml` or `config/keywords_ko.yml`). Each file defines:
- `stems`: keyword stems watched in streaming ASR and LLM predictions
- `numeric_sensitive_stems` + `numeric_scan_window`: stems that trigger extra digit detection immediately after the match
- `llm_model_path`, `context_tokens`, `prompt_template`, and sampling knobs for the horizon predictor

Set `webrtc_vad.enabled` to `false` to fall back to the legacy RMS gate (not recommended except for debugging installs without the `webrtcvad` package).

DAF remains active while either WebRTC VAD detects sustained speech or Vosk continues to emit new partial transcripts. It disengages after `speech_release_ms` of silence (or `daf_max_active_ms`, whichever comes first), which keeps the feedback responsive even in noisy environments.

## Running the Pipeline

1. Activate the virtualenv and ensure the ESP32 is connected over USB.
2. From the `gradi-prediction` directory run:

   ```bash
   python src/main.py --language en         # English mode (default)
   python src/main.py --language ko         # Korean mode
   # add --logging for INFO-level diagnostics, --cpu-only to bypass GPU
   ```

   Use `--cpu-only` when you want to override the default GPU-enabled horizon predictor without editing configs, regardless of language.

3. The host will sync the ESP32 DAF state, stream audio frames over serial, and toggle delayed playback when configured keywords or predicted risky phrases are detected. DAF then releases automatically after speech activity stops or when the maximum active window elapses.

Logs are emitted via `loguru` for ASR partials, LLM predictions, keyword hits, and DAF transitions.

## Troubleshooting

- **Serial cannot open**: Verify group membership (`sudo usermod -aG dialout $USER`) and confirm the `/dev/ttyACM*` path.
- **No audio arriving**: Ensure the ESP32 sketch is flashed and the serial baud matches `esp_serial_baud`.
- **sounddevice errors**: Install PortAudio (above) or keep `audio_source: esp32_serial`.
- **Large model downloads**: Populate `models/` manually if the repo does not include them.

## License

This directory collects the up-to-date integration artifacts for the Gradi predictive speech monitoring prototype.

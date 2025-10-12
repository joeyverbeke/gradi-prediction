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
    pyyaml==6.0.2 loguru==0.7.2 pyserial==3.5
```

If you plan to use the local microphone fallback (`audio_source: local`), install PortAudio system libraries first:

```bash
sudo apt install libportaudio2
```

### Optional: GPU Acceleration for llama.cpp

If you want the horizon predictor to run on your CUDA GPU, rebuild `llama-cpp-python` with CUDA enabled and set `llm_gpu_layers` in `config/keywords.yml` (e.g. `-1` for full auto offload, `0` to stay on CPU).

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

- `models/asr/vosk-model-small-en-us/`
- `models/llm/llama-3.2-1b-q4_k_m.gguf`

Adjust paths in `src/main.py` if you host the assets elsewhere.

## Configuration

Key settings live in `config/audio.yml` (audio pipeline + ESP32 serial) and `config/keywords.yml` (phrases + LLM parameters).

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
esp_serial_port: /dev/ttyACM0
esp_serial_baud: 921600
esp_chunk_samples: 1024
```

Update `esp_serial_port` to match the device exposed when the ESP32 is connected (`ls /dev/ttyACM*` on Ubuntu). If you switch back to host audio capture, change `audio_source` to `local` and configure `input_device`/`output_device`.

## Running the Pipeline

1. Activate the virtualenv and ensure the ESP32 is connected over USB.
2. From the `gradi-prediction` directory run:

   ```bash
   python src/main.py            # add --logging to surface INFO-level diagnostics (default prints errors only)
   ```

3. The host will sync the ESP32 DAF state, stream audio frames over serial, and toggle delayed playback when configured keywords or predicted risky phrases are detected.

Logs are emitted via `loguru` for ASR partials, LLM predictions, keyword hits, and DAF transitions.

## Troubleshooting

- **Serial cannot open**: Verify group membership (`sudo usermod -aG dialout $USER`) and confirm the `/dev/ttyACM*` path.
- **No audio arriving**: Ensure the ESP32 sketch is flashed and the serial baud matches `esp_serial_baud`.
- **sounddevice errors**: Install PortAudio (above) or keep `audio_source: esp32_serial`.
- **Large model downloads**: Populate `models/` manually if the repo does not include them.

## License

This directory collects the up-to-date integration artifacts for the Gradi predictive speech monitoring prototype.

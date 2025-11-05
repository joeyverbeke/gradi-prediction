# Gradi Prediction Speech Monitor

Desktop orchestrator + ESP32-S3 firmware for provoking topic-guided speech, detecting risky phrases, and driving delayed auditory feedback (DAF).

## Quick Start
- Install Python 3.11, ensure `uv` CLI is available, and clone this repo.
- Set up the virtualenv: `uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .`
- Populate `models/` with the required Vosk ASR and llama.cpp GGUF checkpoints (see below).
- Flash `firmware/esp32/esp-daf/esp-daf.ino` to a Seeed XIAO ESP32S3, including the generated prompt headers.
- Run `python src/main.py --language en --logging` (add `--port ttyACM*` to override the serial device).

## Host Environment
- All runtime dependencies live in `pyproject.toml`; `uv pip install -e .` keeps them in sync with the editable code.
- Optional developer extras (ruff, ipython) install via `uv pip install -e .[dev]`.
- For microphone fallback (`audio_source: local`) install PortAudio system libs first: `sudo apt install libportaudio2`.
- Verify WebRTC VAD after setup:
  ```bash
  python - <<'PY'
  import webrtcvad
  print("WebRTC VAD ready:", webrtcvad.__version__)
  PY
  ```
- Need GPU offload? Install a CUDA-enabled `llama-cpp-python` wheel and leave `llm_gpu_layers` at `-1`, or launch with `--cpu-only` to force CPU runs.

## ESP32 Firmware & Prompts
- Open `firmware/esp32/esp-daf/esp-daf.ino` in Arduino IDE / PlatformIO, target the XIAO ESP32S3, and set serial baud to `921600`.
- The sketch depends on `ESP32-SpeexDSP` and `mmwave_for_xiao`; install both through the Arduino Library Manager.
- Presence sensing uses Seeed’s 24 GHz radar (TX→D1, RX→D4). The firmware pauses audio + DAF whenever presence is lost.
- Prompt audio: convert 16 kHz mono PCM files in `tts-prompts/` into headers with `python3 scripts/gen_prompt_header.py --input <raw> --output firmware/esp-daf/<name>.h --symbol <symbol>`, then add the asset to `PROMPT_ASSETS`.

## Models
- English ASR: `models/asr/vosk-model-small-en-us/`
- Korean ASR: `models/asr/vosk-model-small-ko-0.22/`
- English horizon LLM: `models/llm/llama-3.2-1b-q4_k_m.gguf`
- Korean horizon LLM: `models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Adjust paths in the keyword configs if you store checkpoints elsewhere.

## Configuration Overview
- `config/audio.yml`: audio transport, VAD, DAF timing, ESP32 serial port.
- `config/keywords_<lang>.yml`: baseline sensitive stems, numeric stem handling, llama.cpp parameters.
- `config/topics_<lang>.yml`: optional topic rotation definitions (id, prompt asset, per-topic stems). Missing files fall back to the keyword list.
- Launch options: `--logging` upgrades Loguru to INFO, `--cpu-only` disables GPU offload, `--port` overrides `esp_serial_port`.
- Example run commands:
  ```bash
  python src/main.py --language en --logging
  python src/main.py --language ko --port ttyACM1
  ```

## Topic Design
- Each topic entry sets `language`, `asset`, and `stems`; ensure the asset string matches a prompt header compiled into the firmware.
- Keep stems short enough to capture variants (e.g., “trump”, “도널드 트럼프”) while avoiding overly generic hits.
- Numeric-sensitive stems live in the keyword config; per-topic overrides are optional. The host watches the matched stem window for digits to catch account numbers or PINs.

## Troubleshooting
- **Serial unavailable**: confirm `/dev/ttyACM*`, ensure dialout group membership, and match baud to `audio.yml`.
- **No audio**: verify the ESP32 sketch is running, topic prompts compiled, and radar presence reported as active.
- **sounddevice errors**: install PortAudio or stay in ESP serial mode.
- **Large model downloads**: preload `models/` manually; the repo does not ship checkpoints.

## Repository Map
```
gradi-prediction/
├── config/                # YAML runtime settings
├── firmware/esp32/esp-daf # ESP32-S3 Arduino sketch + prompt assets
├── models/                # ASR + LLM checkpoints (not versioned)
└── src/                   # Python host application
```

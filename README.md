# Gradi Prediction Speech Monitor

Desktop orchestrator + ESP32-S3 firmware for provoking topic-guided speech, detecting risky phrases, and driving delayed auditory feedback (DAF).

## Quick Start
- Install Python 3.11, ensure `uv` CLI is available, and clone this repo.
- Set up the virtualenv: `uv venv --python 3.11 && source .venv/bin/activate && uv pip install -e .`
- Populate `models/` with the required Vosk ASR and llama.cpp GGUF checkpoints (see below).
- Flash `firmware/esp32/esp-daf/esp-daf.ino` to a Seeed XIAO ESP32S3, including the generated prompt headers.
- Run `python src/main.py --language ko --logging` (defaults to `/dev/gradi-esp-predict`; add `--port /dev/ttyACM*` to override).

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
- Semantic scorer (optional, Change 3): `all-MiniLM-L6-v2` via `fastembed`, cached under `models/embed/` (≈25 MB). Downloaded automatically on first run; **warm the cache while online** (running `scripts/semantic_eval.py` does this) so an offline installation loads it from cache. Disabled by default — no download needed until you enable it.
- Adjust `horizon.model_path` in the scenario config if you store checkpoints elsewhere.

## Configuration Overview
- `config/audio.yml`: audio transport, VAD (incl. the release-path energy gate), DAF timing, ESP32 serial port.
- `config/scenarios_<lang>.yml`: the consolidated scenario config (replaces the old `keywords_<lang>.yml` + `topics_<lang>.yml` split). Sections:
  - `horizon`: llama.cpp predictor settings (model path, gpu layers, token limits, prompt template).
  - `detection`: numeric follow-up handling (`numeric_scan_window`, `numeric_sensitive_stems`).
  - `semantic`: embedding scenario scorer — `enabled` (ships `false`), `model_name`, `cache_dir`.
  - `scenarios`: prompt round-robin; each has `id`, `prompt` asset, `mode` (`stems`|`semantic`|`both`), `stems`, and an optional per-scenario `semantic` block (`threshold`, `consecutive_hits`, `exemplars`, `contrast`).
- **Deprecation fallback**: if `scenarios_<lang>.yml` is absent, the host still boots by loading the legacy `keywords_<lang>.yml` + `topics_<lang>.yml` pair (logged as a deprecation warning).
- Launch options: `--logging` upgrades Loguru to INFO, `--cpu-only` disables GPU offload, `--port` overrides `esp_serial_port`.
- Example run commands:
  ```bash
  python src/main.py --language en --logging
  python src/main.py --language ko --port /dev/ttyACM1
  ```

## Topic Design
- Each scenario sets `language`, `asset`, `mode`, and `stems`; ensure the asset string matches a prompt header compiled into the firmware.
- **Stems** are the zero-latency fast path (Aho-Corasick substring match). The detector has **no word boundaries**, so avoid bare stems under 5 characters (e.g. use `ai will`, not `ai`) — a short stem matches inside unrelated words.
- **Semantic** (`mode: semantic` or `both`) scores meaning against per-scenario `exemplars` minus `contrast` (hedging/filler) in embedding space, catching paraphrase the stems miss. It runs on a dedicated worker thread and adds no trigger latency. Both production scenarios run `both`.
- Tune scenarios offline with `python scripts/semantic_eval.py` (labeled utterance sets + stem-safety check). Edit `exemplars`/`threshold`/`contrast` in YAML and re-run — no training, no external services. Bias thresholds toward triggering (sensitivity over discrimination).
- Numeric-sensitive stems live in `detection`; per-scenario overrides are optional. The host watches the matched stem window for digits to catch account numbers or PINs.

## Troubleshooting
- **Serial unavailable**: confirm `/dev/gradi-esp-predict` (or the matching `/dev/ttyACM*` device) exists, ensure dialout group membership, and match baud to `audio.yml`.
- **No audio**: verify the ESP32 sketch is running, topic prompts compiled, and radar presence reported as active.
- **sounddevice errors**: install PortAudio or stay in ESP serial mode.
- **Large model downloads**: preload `models/` manually; the repo does not ship checkpoints. If the semantic scorer is enabled, warm its cache while online first (`scripts/semantic_eval.py`) — an offline PC with an empty `models/embed/` disables semantic scoring and falls back to stems (logged), it does not crash.

## Repository Map
```
gradi-prediction/
├── config/                # YAML runtime settings
├── firmware/esp32/esp-daf # ESP32-S3 Arduino sketch + prompt assets
├── models/                # ASR + LLM checkpoints (not versioned)
└── src/                   # Python host application
```

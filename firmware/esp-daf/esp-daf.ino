#include <Arduino.h>
#include <cstring>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <pgmspace.h>
#include "driver/i2s.h"
#include <ESP32-SpeexDSP.h>
#include <mmwave_for_xiao.h>
#include "kr_prompt_brag.h"
#include "kr_prompt_politics.h"
#include "en_prompt_politics.h"
#include "en_prompt_brag.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===== Configuration =====
static constexpr int SAMPLE_RATE = 16000;
static constexpr int CHUNK_SAMPLES = 1024;
static constexpr int SERIAL_BAUD = 921600;
static constexpr int DELAY_MS = 175;
static constexpr int DELAY_SAMPLES = SAMPLE_RATE * DELAY_MS / 1000; // 2800
static constexpr int RING_CAPACITY = 4096; // Must exceed delay samples
static constexpr float MAX_OUTPUT_DBFS = -18.0f; // Adjust to set absolute playback ceiling

// I2S microphone (ICS43434) pins on XIAO ESP32S3
static constexpr int PIN_MIC_BCLK = 3;  // D2 -> GPIO3
static constexpr int PIN_MIC_WS = 1;    // D0 -> GPIO1 (LRCL)
static constexpr int PIN_MIC_SD = 6;    // D5 -> GPIO6 (DOUT)
static constexpr int PIN_MIC_SEL = 4;   // D3 -> GPIO4 (SEL, LOW selects left)

// I2S speaker pins (MAX98357A)
static constexpr int PIN_SPK_BCLK = 8;  // D9 -> GPIO8
static constexpr int PIN_SPK_WS = 7;    // D8 -> GPIO7
static constexpr int PIN_SPK_SDOUT = 9; // D10 -> GPIO9
static constexpr int PIN_SPK_ENABLE = 44; // D7 -> GPIO44

// 24 GHz mmWave radar pins (HardwareSerial1)
static constexpr int PIN_RADAR_RX = 2;  // D1 -> GPIO2 (radar TX -> ESP RX)
static constexpr int PIN_RADAR_TX = 5;  // D4 -> GPIO5 (radar RX <- ESP TX)
static constexpr int RADAR_BAUD = 256000;
static constexpr int PRESENCE_THRESHOLD_MM = 30;            // 0.3 m threshold to detect someone
static constexpr uint16_t POLL_DELAY_NO_PRESENCE_MS = 20;   // Fast polling when idle
static constexpr uint16_t POLL_DELAY_PRESENCE_MS = 100;     // Slow polling once latched
static constexpr uint16_t PRESENCE_CLEAR_DELAY_MS = 1500;   // Require absence this long before disabling

static constexpr i2s_port_t I2S_PORT_MIC = I2S_NUM_1;
static constexpr i2s_port_t I2S_PORT_SPK = I2S_NUM_0;

// ===== State =====
static bool dafEnabled = false;
static int16_t ringBuffer[RING_CAPACITY];
static size_t ringWriteIndex = 0;
static size_t ringFilled = 0;

static int16_t streamBuffer[CHUNK_SAMPLES];
static size_t streamSamples = 0;
static uint8_t playbackBuffer[256 * sizeof(int16_t)]; // 256 mono samples per push
static size_t playbackSamples = 0;

static char commandBuffer[32];
static size_t commandIndex = 0;

// SpeexDSP preprocessor state
static constexpr int DSP_FRAME_SAMPLES = 160; // 10ms @ 16kHz
static SpeexPreprocessState *speexState = nullptr;
static SpeexEchoState *speexEchoState = nullptr;
static int16_t micFrame[DSP_FRAME_SAMPLES];
static int16_t aecFrame[DSP_FRAME_SAMPLES];
static int16_t playbackFrame[DSP_FRAME_SAMPLES];
static size_t dspIndex = 0;

// Simple biquad filter chain for playback conditioning
struct Biquad {
  float b0, b1, b2, a1, a2;
  float z1, z2;
};

static Biquad hpFilter;    // 850 Hz high-pass
static Biquad notchFilter; // 6.3 kHz notch
static Biquad lpFilter;    // 6.0 kHz low-pass

static const float MAX_OUTPUT_SCALE = powf(10.0f, MAX_OUTPUT_DBFS / 20.0f);
static const float LIMIT_CEILING = MAX_OUTPUT_SCALE * 32767.0f;
static const float LIMIT_THRESHOLD = LIMIT_CEILING * 0.75f;
static const float PLAYBACK_PRE_GAIN = MAX_OUTPUT_SCALE;

// Presence gate state
static HardwareSerial radarSerial(1);
static Seeed_HSP24 radarDevice(radarSerial, Serial);
static bool presenceActive = false;
static bool presenceInitialized = false;
static uint32_t lastPresencePollMs = 0;
static uint32_t lastPresenceDetectedMs = 0;
static bool presenceDebug = false;
static uint32_t lastPresenceDebugMs = 0;

struct PromptAsset {
  const char *language;
  const char *topic;
  const int16_t *samples;
  size_t sampleCount;
};

static constexpr PromptAsset PROMPT_ASSETS[] = {
  {"KR", "POLITICS", kr_prompt_politics, kr_prompt_politics_length},
  {"KR", "BRAG",     kr_prompt_brag,     kr_prompt_brag_length},
  {"EN", "POLITICS", en_prompt_politics, en_prompt_politics_length},
  {"EN", "BRAG",     en_prompt_brag,     en_prompt_brag_length},
};
static constexpr size_t PROMPT_ASSET_COUNT = sizeof(PROMPT_ASSETS) / sizeof(PROMPT_ASSETS[0]);
static constexpr size_t PROMPT_SLICE_SAMPLES = 256;

struct PromptPlayer {
  const PromptAsset *asset = nullptr;
  size_t index = 0;
  bool active = false;

  void stop() {
    asset = nullptr;
    index = 0;
    active = false;
  }
};

static PromptPlayer promptPlayer;
static void sendLine(const char *line);
static void flushPlaybackBuffer();
static inline void appendPlaybackSample(int16_t sample);
static void suspendAudioPipeline();

static bool equalsIgnoreCase(const char *a, const char *b) {
  if (!a || !b) {
    return false;
  }
  while (*a && *b) {
    if (toupper(static_cast<unsigned char>(*a)) != toupper(static_cast<unsigned char>(*b))) {
      return false;
    }
    ++a;
    ++b;
  }
  return *a == '\0' && *b == '\0';
}

static const PromptAsset *findPromptAsset(const char *language, const char *topic) {
  for (size_t i = 0; i < PROMPT_ASSET_COUNT; ++i) {
    const PromptAsset &asset = PROMPT_ASSETS[i];
    if (equalsIgnoreCase(language, asset.language) && equalsIgnoreCase(topic, asset.topic)) {
      return &asset;
    }
  }
  return nullptr;
}

static void notifyPromptStatus(const char *status, const char *detail = nullptr) {
  if (detail && detail[0] != '\0') {
    char line[96];
    snprintf(line, sizeof(line), "%s %s", status, detail);
    sendLine(line);
  } else {
    sendLine(status);
  }
}

static void finishPromptPlayback(const char *status) {
  const PromptAsset *asset = promptPlayer.asset;
  promptPlayer.stop();
  flushPlaybackBuffer();

  if (asset) {
    char detail[32];
    snprintf(detail, sizeof(detail), "%s %s", asset->language, asset->topic);
    notifyPromptStatus(status, detail);
  } else {
    notifyPromptStatus(status);
  }
}

static void abortPromptPlayback(const char *reason) {
  if (!promptPlayer.active) {
    return;
  }
  const PromptAsset *asset = promptPlayer.asset;
  promptPlayer.stop();
  flushPlaybackBuffer();
  char detail[64] = {0};
  if (asset) {
    snprintf(detail, sizeof(detail), "%s %s", asset->language, asset->topic);
  }
  if (reason && reason[0] != '\0') {
    char message[96];
    if (detail[0] != '\0') {
      snprintf(message, sizeof(message), "%s | %s", detail, reason);
    } else {
      snprintf(message, sizeof(message), "%s", reason);
    }
    notifyPromptStatus("PROMPT ABORT", message);
  } else if (detail[0] != '\0') {
    notifyPromptStatus("PROMPT ABORT", detail);
  } else {
    notifyPromptStatus("PROMPT ABORT");
  }
}

static void processPromptPlayback() {
  if (!promptPlayer.active || !promptPlayer.asset) {
    return;
  }

  size_t produced = 0;
  while (produced < PROMPT_SLICE_SAMPLES && promptPlayer.index < promptPlayer.asset->sampleCount) {
    int16_t sample = static_cast<int16_t>(pgm_read_word(&promptPlayer.asset->samples[promptPlayer.index++]));
    appendPlaybackSample(sample);
    ++produced;
  }

  if (promptPlayer.index >= promptPlayer.asset->sampleCount) {
    finishPromptPlayback("PROMPT DONE");
  } else {
    flushPlaybackBuffer();
  }
}

static bool startPromptPlayback(const PromptAsset *asset) {
  if (!asset || asset->sampleCount == 0) {
    return false;
  }

  suspendAudioPipeline();
  promptPlayer.asset = asset;
  promptPlayer.index = 0;
  promptPlayer.active = true;
  char detail[32];
  snprintf(detail, sizeof(detail), "%s %s", asset->language, asset->topic);
  notifyPromptStatus("PROMPT OK", detail);
  return true;
}

static Biquad makeBiquad(float b0, float b1, float b2, float a0, float a1, float a2) {
  Biquad biq{};
  biq.b0 = b0 / a0;
  biq.b1 = b1 / a0;
  biq.b2 = b2 / a0;
  biq.a1 = a1 / a0;
  biq.a2 = a2 / a0;
  biq.z1 = 0.0f;
  biq.z2 = 0.0f;
  return biq;
}

static Biquad makeHighpass(float freq, float q) {
  const float w0 = 2.0f * M_PI * freq / SAMPLE_RATE;
  const float cosw0 = cosf(w0);
  const float sinw0 = sinf(w0);
  const float alpha = sinw0 / (2.0f * q);

  const float b0 = (1.0f + cosw0) / 2.0f;
  const float b1 = -(1.0f + cosw0);
  const float b2 = (1.0f + cosw0) / 2.0f;
  const float a0 = 1.0f + alpha;
  const float a1 = -2.0f * cosw0;
  const float a2 = 1.0f - alpha;

  return makeBiquad(b0, b1, b2, a0, a1, a2);
}

static Biquad makeLowpass(float freq, float q) {
  const float w0 = 2.0f * M_PI * freq / SAMPLE_RATE;
  const float cosw0 = cosf(w0);
  const float sinw0 = sinf(w0);
  const float alpha = sinw0 / (2.0f * q);

  const float b0 = (1.0f - cosw0) / 2.0f;
  const float b1 = 1.0f - cosw0;
  const float b2 = (1.0f - cosw0) / 2.0f;
  const float a0 = 1.0f + alpha;
  const float a1 = -2.0f * cosw0;
  const float a2 = 1.0f - alpha;

  return makeBiquad(b0, b1, b2, a0, a1, a2);
}

static Biquad makeNotch(float freq, float q) {
  const float w0 = 2.0f * M_PI * freq / SAMPLE_RATE;
  const float cosw0 = cosf(w0);
  const float sinw0 = sinf(w0);
  const float alpha = sinw0 / (2.0f * q);

  const float b0 = 1.0f;
  const float b1 = -2.0f * cosw0;
  const float b2 = 1.0f;
  const float a0 = 1.0f + alpha;
  const float a1 = -2.0f * cosw0;
  const float a2 = 1.0f - alpha;

  return makeBiquad(b0, b1, b2, a0, a1, a2);
}

static inline float processBiquad(Biquad &biq, float in) {
  const float out = biq.b0 * in + biq.z1;
  biq.z1 = biq.b1 * in - biq.a1 * out + biq.z2;
  biq.z2 = biq.b2 * in - biq.a2 * out;
  return out;
}

static void configureFilterChain() {
  hpFilter = makeHighpass(850.0f, 0.707f);
  notchFilter = makeNotch(6300.0f, 20.0f);
  lpFilter = makeLowpass(6000.0f, 0.707f);
}

struct LookaheadLimiter {
  static constexpr size_t LOOKAHEAD = 32; // ~2 ms at 16 kHz
  static constexpr float RELEASE_TIME_SEC = 0.1f; // 100 ms
  static constexpr size_t BUFFER_CAP = LOOKAHEAD + DSP_FRAME_SAMPLES;

  struct Slot {
    float sample;
    float gain;
  };

  Slot buffer[BUFFER_CAP];
  size_t head = 0;
  size_t tail = 0;
  size_t count = 0;
  float currentGain = 1.0f;
  float releaseCoeff = 0.0f;

  void init() {
    const float releaseSamples = RELEASE_TIME_SEC * static_cast<float>(SAMPLE_RATE);
    releaseCoeff = releaseSamples > 0.0f ? expf(-1.0f / releaseSamples) : 0.0f;
    reset();
  }

  void reset() {
    head = 0;
    tail = 0;
    count = 0;
    currentGain = 1.0f;
    for (size_t i = 0; i < LOOKAHEAD; ++i) {
      pushSilence();
    }
  }

  void pushSilence() {
    buffer[tail].sample = 0.0f;
    buffer[tail].gain = 1.0f;
    tail = (tail + 1) % BUFFER_CAP;
    ++count;
  }

  bool processSample(float sample, float &outSample) {
    buffer[tail].sample = sample;
    buffer[tail].gain = 1.0f;
    tail = (tail + 1) % BUFFER_CAP;
    if (count < BUFFER_CAP) {
      ++count;
    } else {
      head = (head + 1) % BUFFER_CAP;
    }

    float maxAbs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      const size_t idx = (head + i) % BUFFER_CAP;
      const float absVal = fabsf(buffer[idx].sample);
      if (absVal > maxAbs) {
        maxAbs = absVal;
      }
    }

    float targetGain = 1.0f;
    if (maxAbs > LIMIT_THRESHOLD && maxAbs > 0.0f) {
      const float desired = LIMIT_CEILING / maxAbs;
      if (desired < targetGain) {
        targetGain = desired;
      }
    }

    if (targetGain < currentGain) {
      currentGain = targetGain;
    } else {
      const float interp = 1.0f - releaseCoeff;
      currentGain += (targetGain - currentGain) * interp;
    }

    if (currentGain > 1.0f) {
      currentGain = 1.0f;
    } else if (currentGain < 0.0f) {
      currentGain = 0.0f;
    }

    const size_t latestIdx = (tail + BUFFER_CAP - 1) % BUFFER_CAP;
    buffer[latestIdx].gain = currentGain;

    if (count > LOOKAHEAD) {
      const Slot slot = buffer[head];
      head = (head + 1) % BUFFER_CAP;
      --count;
      outSample = slot.sample * slot.gain;
      return true;
    }

    return false;
  }

};

static LookaheadLimiter playbackLimiter;

// ===== Helpers =====
static void sendLine(const char *line) {
  Serial.write(line);
  Serial.write('\n');
}

void publishPresence() {
  sendLine(presenceActive ? "PRESENCE ON" : "PRESENCE OFF");
}

void publishState() {
  sendLine(dafEnabled ? "STATE ON" : "STATE OFF");
}

void handleCommand(const char *cmd) {
  if (strcmp(cmd, "STATE?") == 0) {
    publishState();
    publishPresence();
    return;
  }

  if (strcmp(cmd, "PRESENCE?") == 0) {
    publishPresence();
    return;
  }

  if (strncmp(cmd, "DAF ", 4) == 0) {
    const char *arg = cmd + 4;
    if (strcmp(arg, "ON") == 0) {
      dafEnabled = true;
      publishState();
    } else if (strcmp(arg, "OFF") == 0) {
      dafEnabled = false;
      publishState();
    } else {
      sendLine("LOG Unknown DAF command");
    }
    return;
  }

  if (strncmp(cmd, "PROMPT ", 7) == 0) {
    if (promptPlayer.active) {
      notifyPromptStatus("PROMPT ERR BUSY");
      return;
    }

    char language[8] = {0};
    char topic[24] = {0};
    if (sscanf(cmd + 7, "%7s %23s", language, topic) != 2) {
      notifyPromptStatus("PROMPT ERR ARG");
      return;
    }

    const PromptAsset *asset = findPromptAsset(language, topic);
    if (!asset) {
      notifyPromptStatus("PROMPT ERR UNKNOWN");
      return;
    }

    if (!presenceActive) {
      notifyPromptStatus("PROMPT ERR NOPRESENCE");
      return;
    }

    if (!startPromptPlayback(asset)) {
      notifyPromptStatus("PROMPT ERR START");
      return;
    }

    char logLine[64];
    snprintf(logLine, sizeof(logLine), "LOG Prompt %s/%s", asset->language, asset->topic);
    sendLine(logLine);
    return;
  }

  if (strncmp(cmd, "DEBUG ", 6) == 0) {
    const char *arg = cmd + 6;
    if (strncmp(arg, "PRESENCE ", 9) == 0) {
      const char *state = arg + 9;
      if (strcmp(state, "ON") == 0) {
        presenceDebug = true;
        sendLine("LOG Presence debug ON");
        return;
      }
      if (strcmp(state, "OFF") == 0) {
        presenceDebug = false;
        sendLine("LOG Presence debug OFF");
        return;
      }
    }
    sendLine("LOG Unknown DEBUG command");
    return;
  }

  sendLine("LOG Unknown command");
}

void checkSerialCommands() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      if (commandIndex > 0) {
        commandBuffer[commandIndex] = '\0';
        handleCommand(commandBuffer);
        commandIndex = 0;
      }
    } else if (commandIndex < sizeof(commandBuffer) - 1) {
      commandBuffer[commandIndex++] = c;
    }
  }
}

static void suspendAudioPipeline() {
  ringWriteIndex = 0;
  ringFilled = 0;
  streamSamples = 0;
  playbackSamples = 0;
  dspIndex = 0;
  memset(micFrame, 0, sizeof(micFrame));
  memset(aecFrame, 0, sizeof(aecFrame));
  memset(playbackFrame, 0, sizeof(playbackFrame));
  playbackLimiter.reset();
  configureFilterChain();
  if (speexEchoState) {
    speex_echo_state_reset(speexEchoState);
  }
  setupSpeexPreprocessor();
  i2s_zero_dma_buffer(I2S_PORT_MIC);
  i2s_zero_dma_buffer(I2S_PORT_SPK);
}

void setPresenceState(bool detected, int distanceMm) {
  presenceActive = detected;
  presenceInitialized = true;
  publishPresence();
  char logBuf[64];
  snprintf(
    logBuf,
    sizeof(logBuf),
    "LOG Presence %s (%dmm)",
    detected ? "ON" : "OFF",
    distanceMm
  );
  sendLine(logBuf);

  if (!detected) {
    if (promptPlayer.active) {
      abortPromptPlayback("PRESENCE");
    }
    if (dafEnabled) {
      dafEnabled = false;
      publishState();
    }
    suspendAudioPipeline();
  }
}

void updatePresenceGate() {
  const uint32_t now = millis();
  const uint16_t pollDelay = presenceActive ? POLL_DELAY_PRESENCE_MS : POLL_DELAY_NO_PRESENCE_MS;
  if (now - lastPresencePollMs < pollDelay) {
    return;
  }
  lastPresencePollMs = now;

  const auto radarStatus = radarDevice.getStatus();
  const int currentDistance = radarStatus.distance;

  if (presenceDebug) {
    const uint32_t dbgNow = now;
    if (dbgNow - lastPresenceDebugMs >= 200) {
      lastPresenceDebugMs = dbgNow;
      char dbgLine[64];
      snprintf(
        dbgLine,
        sizeof(dbgLine),
        "LOG Radar distance=%dmm target=%d",
        currentDistance,
        static_cast<int>(radarStatus.targetStatus)
      );
      sendLine(dbgLine);
    }
  }

  if (currentDistance == -1) {
    return;
  }

  const bool detected = currentDistance <= PRESENCE_THRESHOLD_MM;
  if (detected) {
    lastPresenceDetectedMs = now;
    if (!presenceInitialized || !presenceActive) {
      setPresenceState(true, currentDistance);
    }
  } else {
    if (!presenceInitialized) {
      setPresenceState(false, currentDistance);
      lastPresenceDetectedMs = 0;
      return;
    }

    if (presenceActive) {
      const bool clearReady =
        (lastPresenceDetectedMs == 0) || ((now - lastPresenceDetectedMs) >= PRESENCE_CLEAR_DELAY_MS);
      if (clearReady) {
        setPresenceState(false, currentDistance);
        lastPresenceDetectedMs = 0;
      }
    }
  }
}

void setupI2SMicrophone() {
  i2s_config_t config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0
  };

  i2s_pin_config_t pinConfig = {
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = PIN_MIC_BCLK,
      .ws_io_num = PIN_MIC_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = PIN_MIC_SD
  };

  ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT_MIC, &config, 0, nullptr));
  ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT_MIC, &pinConfig));
  ESP_ERROR_CHECK(i2s_set_clk(I2S_PORT_MIC, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO));
}

void setupI2SSpeaker() {
  i2s_config_t config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = true,
      .fixed_mclk = 0
  };

  i2s_pin_config_t pinConfig = {
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = PIN_SPK_BCLK,
      .ws_io_num = PIN_SPK_WS,
      .data_out_num = PIN_SPK_SDOUT,
      .data_in_num = I2S_PIN_NO_CHANGE
  };

  ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT_SPK, &config, 0, nullptr));
  ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT_SPK, &pinConfig));
  ESP_ERROR_CHECK(i2s_set_clk(I2S_PORT_SPK, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO));
  ESP_ERROR_CHECK(i2s_zero_dma_buffer(I2S_PORT_SPK));
}

void setupSpeexPreprocessor() {
  if (speexState != nullptr) {
    speex_preprocess_state_destroy(speexState);
    speexState = nullptr;
  }

  speexState = speex_preprocess_state_init(DSP_FRAME_SAMPLES, SAMPLE_RATE);
  if (!speexState) {
    sendLine("LOG Failed to init Speex preprocessor");
    return;
  }

  int enable = 1;
  int disable = 0;
  int noiseSuppressDb = -36;   // Strong default denoise; adjust if necessary

  speex_preprocess_ctl(speexState, SPEEX_PREPROCESS_SET_DENOISE, &enable);
  speex_preprocess_ctl(speexState, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &noiseSuppressDb);
  speex_preprocess_ctl(speexState, SPEEX_PREPROCESS_SET_AGC, &disable);
  speex_preprocess_ctl(speexState, SPEEX_PREPROCESS_SET_VAD, &disable);
  if (speexEchoState) {
    speex_preprocess_ctl(speexState, SPEEX_PREPROCESS_SET_ECHO_STATE, speexEchoState);
  }
}

void setupSpeexEcho() {
  if (speexEchoState != nullptr) {
    speex_echo_state_destroy(speexEchoState);
    speexEchoState = nullptr;
  }

  speexEchoState = speex_echo_state_init(DSP_FRAME_SAMPLES, 1024);
  if (!speexEchoState) {
    sendLine("LOG Failed to init Speex echo canceller");
    return;
  }

  int sampleRate = SAMPLE_RATE;
  speex_echo_ctl(speexEchoState, SPEEX_ECHO_SET_SAMPLING_RATE, &sampleRate);
}

void setupRadar() {
  radarSerial.begin(RADAR_BAUD, SERIAL_8N1, PIN_RADAR_RX, PIN_RADAR_TX);
  sendLine("LOG Radar UART ready");
  publishPresence();
}

void sendAudioChunk() {
  const size_t byteCount = streamSamples * sizeof(int16_t);
  char header[24];
  snprintf(header, sizeof(header), "AUDIO %zu", byteCount);
  sendLine(header);
  Serial.write(reinterpret_cast<uint8_t *>(streamBuffer), byteCount);
  streamSamples = 0;
}

static void flushPlaybackBuffer() {
  if (playbackSamples == 0) {
    return;
  }
  size_t bytes = playbackSamples * sizeof(int16_t);
  size_t written = 0;
  i2s_write(I2S_PORT_SPK, playbackBuffer, bytes, &written, portMAX_DELAY);
  playbackSamples = 0;
}

static inline void appendPlaybackSample(int16_t sample) {
  reinterpret_cast<int16_t *>(playbackBuffer)[playbackSamples++] = sample;
  if (playbackSamples == (sizeof(playbackBuffer) / sizeof(int16_t))) {
    flushPlaybackBuffer();
  }
}

static void generatePlaybackFrame(int16_t *outFrame, size_t writeIndexSnapshot, size_t ringFilledSnapshot) {
  const size_t delaySamples = DELAY_SAMPLES;
  size_t produced = 0;
  size_t appended = 0;
  const size_t readableSamples = (ringFilledSnapshot > delaySamples)
                                   ? (ringFilledSnapshot - delaySamples)
                                   : 0;

  for (size_t i = 0; i < DSP_FRAME_SAMPLES; ++i) {
    int16_t delayedSample = 0;
    if (dafEnabled && ringFilledSnapshot > delaySamples && i < readableSamples) {
      size_t readIndex = (writeIndexSnapshot + RING_CAPACITY - delaySamples + i) % RING_CAPACITY;
      delayedSample = ringBuffer[readIndex];
    }

    float processed = static_cast<float>(delayedSample) * PLAYBACK_PRE_GAIN;
    processed = processBiquad(hpFilter, processed);
    processed = processBiquad(notchFilter, processed);
    processed = processBiquad(lpFilter, processed);

    float limited = 0.0f;
    if (playbackLimiter.processSample(processed, limited)) {
      if (limited > LIMIT_CEILING) {
        limited = LIMIT_CEILING;
      } else if (limited < -LIMIT_CEILING) {
        limited = -LIMIT_CEILING;
      }
      const int16_t outSample = static_cast<int16_t>(std::round(limited));
      if (produced < DSP_FRAME_SAMPLES) {
        outFrame[produced++] = outSample;
      }
      appendPlaybackSample(outSample);
      ++appended;
    }
  }

  while (produced < DSP_FRAME_SAMPLES) {
    float limited = 0.0f;
    if (!playbackLimiter.processSample(0.0f, limited)) {
      break;
    }
    if (limited > LIMIT_CEILING) {
      limited = LIMIT_CEILING;
    } else if (limited < -LIMIT_CEILING) {
      limited = -LIMIT_CEILING;
    }
    const int16_t outSample = static_cast<int16_t>(std::round(limited));
    outFrame[produced++] = outSample;
    appendPlaybackSample(outSample);
    ++appended;
  }

  while (produced < DSP_FRAME_SAMPLES) {
    outFrame[produced++] = 0;
    appendPlaybackSample(0);
    ++appended;
  }

  (void)appended; // appended should equal DSP_FRAME_SAMPLES; kept to avoid unused warning
}

static void processAudioFrame() {
  const size_t writeIndexSnapshot = ringWriteIndex;
  const size_t ringFilledSnapshot = ringFilled;

  generatePlaybackFrame(playbackFrame, writeIndexSnapshot, ringFilledSnapshot);

  if (speexEchoState) {
    speex_echo_cancellation(speexEchoState, micFrame, playbackFrame, aecFrame);
  } else {
    memcpy(aecFrame, micFrame, sizeof(aecFrame));
  }

  if (speexState) {
    speex_preprocess_run(speexState, aecFrame);
  }

  for (size_t j = 0; j < DSP_FRAME_SAMPLES; ++j) {
    const int16_t cleaned = aecFrame[j];
    ringBuffer[ringWriteIndex] = cleaned;
    ringWriteIndex = (ringWriteIndex + 1) % RING_CAPACITY;
    if (ringFilled < RING_CAPACITY) {
      ++ringFilled;
    }

    streamBuffer[streamSamples++] = micFrame[j];
    if (streamSamples == CHUNK_SAMPLES) {
      sendAudioChunk();
    }
  }
}

void processMicFrames(const int32_t *frames, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    int32_t sample32 = frames[i];
    int16_t sample = static_cast<int16_t>(sample32 >> 12);

    micFrame[dspIndex++] = sample;

    if (dspIndex == DSP_FRAME_SAMPLES) {
      processAudioFrame();
      dspIndex = 0;
    }
  }
}

void setup() {
  Serial.setRxBufferSize(32768);
  Serial.begin(SERIAL_BAUD);
  pinMode(PIN_MIC_SEL, OUTPUT);
  digitalWrite(PIN_MIC_SEL, LOW); // select left channel
  pinMode(PIN_SPK_ENABLE, OUTPUT);
  digitalWrite(PIN_SPK_ENABLE, HIGH);

  configureFilterChain();
  playbackLimiter.init();

  setupI2SMicrophone();
  setupI2SSpeaker();
  setupSpeexEcho();
  setupSpeexPreprocessor();
  setupRadar();

  sendLine("LOG DAF firmware booted");
  publishState();
}

void loop() {
  updatePresenceGate();
  checkSerialCommands();

  if (promptPlayer.active) {
    processPromptPlayback();
    delay(1);
    return;
  }

  if (!presenceActive) {
    delay(5);
    return;
  }

  static int32_t micBuffer[256];
  size_t bytesRead = 0;
  esp_err_t err = i2s_read(I2S_PORT_MIC, micBuffer, sizeof(micBuffer), &bytesRead, 10 / portTICK_PERIOD_MS);
  if (err == ESP_OK && bytesRead > 0) {
    size_t frames = bytesRead / sizeof(int32_t);
    processMicFrames(micBuffer, frames);
  } else {
    // If no audio arrived, yield a little to avoid watchdog complaints
    delay(1);
  }

  if (streamSamples > 0 && streamSamples < CHUNK_SAMPLES) {
    // Optionally flush partial chunk after some time? we'll rely on continuous capture
  }

  if (playbackSamples > 0) {
    flushPlaybackBuffer();
  }
}

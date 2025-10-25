#include <Arduino.h>
#include <cstring>
#include <cmath>
#include "driver/i2s.h"
#include <ESP32-SpeexDSP.h>

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
static int16_t dspFrame[DSP_FRAME_SAMPLES];
static int16_t rawFrame[DSP_FRAME_SAMPLES];
static size_t dspIndex = 0;

// Simple biquad filter chain for playback conditioning
struct Biquad {
  float b0, b1, b2, a1, a2;
  float z1, z2;
};

static Biquad hpFilter;    // 850 Hz high-pass
static Biquad presenceEQ;  // ~2.5 kHz peaking EQ
static Biquad lpFilter;    // 6.8 kHz low-pass

static constexpr float PLAYBACK_PRE_GAIN = 0.5f;          // -6 dB
static constexpr float PLAYBACK_LIMIT = 0.707f * 32767.0f; // -3 dBFS ceiling

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

static Biquad makePeaking(float freq, float q, float gainDb) {
  const float w0 = 2.0f * M_PI * freq / SAMPLE_RATE;
  const float cosw0 = cosf(w0);
  const float sinw0 = sinf(w0);
  const float alpha = sinw0 / (2.0f * q);
  const float A = powf(10.0f, gainDb / 40.0f);

  const float b0 = 1.0f + alpha * A;
  const float b1 = -2.0f * cosw0;
  const float b2 = 1.0f - alpha * A;
  const float a0 = 1.0f + alpha / A;
  const float a1 = -2.0f * cosw0;
  const float a2 = 1.0f - alpha / A;

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
  presenceEQ = makePeaking(2500.0f, 0.7f, 2.5f);
  lpFilter = makeLowpass(6800.0f, 0.707f);
}

// ===== Helpers =====
void sendLine(const char *line) {
  Serial.write(line);
  Serial.write('\n');
}

void publishState() {
  sendLine(dafEnabled ? "STATE ON" : "STATE OFF");
}

void handleCommand(const char *cmd) {
  if (strcmp(cmd, "STATE?") == 0) {
    publishState();
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
}

void sendAudioChunk() {
  const size_t byteCount = streamSamples * sizeof(int16_t);
  char header[24];
  snprintf(header, sizeof(header), "AUDIO %zu", byteCount);
  sendLine(header);
  Serial.write(reinterpret_cast<uint8_t *>(streamBuffer), byteCount);
  streamSamples = 0;
}

void flushPlaybackBuffer() {
  if (playbackSamples == 0) {
    return;
  }
  size_t bytes = playbackSamples * sizeof(int16_t);
  size_t written = 0;
  i2s_write(I2S_PORT_SPK, playbackBuffer, bytes, &written, portMAX_DELAY);
  playbackSamples = 0;
}

void processMicFrames(const int32_t *frames, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    int32_t sample32 = frames[i];
    int16_t sample = static_cast<int16_t>(sample32 >> 12);

    rawFrame[dspIndex] = sample;
    dspFrame[dspIndex++] = sample;

    if (dspIndex == DSP_FRAME_SAMPLES) {
      if (speexState) {
        speex_preprocess_run(speexState, dspFrame);
      }

      for (size_t j = 0; j < DSP_FRAME_SAMPLES; ++j) {
        int16_t cleaned = dspFrame[j];
        int16_t raw = rawFrame[j];

        ringBuffer[ringWriteIndex] = cleaned;
        ringWriteIndex = (ringWriteIndex + 1) % RING_CAPACITY;
        if (ringFilled < RING_CAPACITY) {
          ringFilled++;
        }

        streamBuffer[streamSamples++] = raw;
        if (streamSamples == CHUNK_SAMPLES) {
          sendAudioChunk();
        }

        int16_t playbackSample = 0;
        if (dafEnabled && ringFilled > DELAY_SAMPLES) {
          size_t readIndex = (ringWriteIndex + RING_CAPACITY - DELAY_SAMPLES) % RING_CAPACITY;
          playbackSample = ringBuffer[readIndex];
        }

        float processed = static_cast<float>(playbackSample) * PLAYBACK_PRE_GAIN;
        processed = processBiquad(hpFilter, processed);
        processed = processBiquad(presenceEQ, processed);
        processed = processBiquad(lpFilter, processed);

        if (processed > PLAYBACK_LIMIT) {
          processed = PLAYBACK_LIMIT;
        } else if (processed < -PLAYBACK_LIMIT) {
          processed = -PLAYBACK_LIMIT;
        }
        playbackSample = static_cast<int16_t>(processed);

        reinterpret_cast<int16_t *>(playbackBuffer)[playbackSamples++] = playbackSample;
        if (playbackSamples == (sizeof(playbackBuffer) / sizeof(int16_t))) {
          flushPlaybackBuffer();
        }
      }

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

  setupI2SMicrophone();
  setupI2SSpeaker();
  setupSpeexPreprocessor();
  configureFilterChain();

  sendLine("LOG DAF firmware booted");
  publishState();
}

void loop() {
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

  checkSerialCommands();

  if (streamSamples > 0 && streamSamples < CHUNK_SAMPLES) {
    // Optionally flush partial chunk after some time? we'll rely on continuous capture
  }

  if (playbackSamples > 0) {
    flushPlaybackBuffer();
  }
}

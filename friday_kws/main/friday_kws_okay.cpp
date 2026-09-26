#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include "driver/i2s_std.h"
#include "esp_dsp.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "custom_model.h"
#include "dsp_weights.h"

// ---------------- Pins ----------------
#define I2S_WS   15
#define I2S_SD   17
#define I2S_SCK  16
#define I2S_PORT I2S_NUM_0
// INMP441 L/R pin: tie to GND -> LEFT, tie to 3V3 -> RIGHT. Do NOT leave it floating.
#define MIC_SLOT I2S_STD_SLOT_LEFT

// ---------------- Must match train.py ----------------
#define SAMPLE_RATE   16000
#define AUDIO_LEN     16000
#define WINDOW_SIZE   480
#define STRIDE        320
#define FFT_N         512
#define NUM_MEL_BINS  40
#define NUM_FRAMES    49
#define RMS_GATE      0.008f
#define STD_GATE      0.15f

// ---------------- Streaming / detection ----------------
#define HOP_FRAMES     6
#define HOP_SAMPLES    (HOP_FRAMES * STRIDE)   // 1920 samples = 120 ms
#define HOP_MS         120
#define WARMUP_HOPS    2
#define THRESHOLD      0.75f
#define HITS_REQUIRED  2
#define COOLDOWN_HOPS  4
#define INPUT_GAIN     1.0f
#define VERBOSE        1

// ---------------- TFLite ----------------
static const int kTensorArenaSize = 80 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// ---------------- Buffers ----------------
alignas(16) static float hann_win[WINDOW_SIZE];
alignas(16) static float fft_buf[FFT_N * 2];
static float mag_spec[FFT_N / 2 + 1];
static float log_mel[NUM_FRAMES][NUM_MEL_BINS];   // cached; only HOP_FRAMES new rows computed per hop
static int16_t audio_buf[AUDIO_LEN];
static int32_t i2s_raw[HOP_SAMPLES];

// Non-zero span of each triangular mel filter (skips ~95% of the 40x257 MACs and the flash reads)
static uint16_t mel_lo[NUM_MEL_BINS], mel_hi[NUM_MEL_BINS];
static int mag_lo, mag_hi;

static i2s_chan_handle_t rx_handle;
static volatile uint32_t i2s_overflow_count = 0;

static bool IRAM_ATTR on_i2s_overflow(i2s_chan_handle_t handle, i2s_event_data_t* event, void* user_ctx) {
    i2s_overflow_count++;
    return false;
}

void init_i2s() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 12;
    chan_cfg.dma_frame_num = 480;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)I2S_SCK,
            .ws   = (gpio_num_t)I2S_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)I2S_SD,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    std_cfg.slot_cfg.slot_mask = MIC_SLOT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));

    i2s_event_callbacks_t cbs = {};
    cbs.on_recv_q_ovf = on_i2s_overflow;
    ESP_ERROR_CHECK(i2s_channel_register_event_callback(rx_handle, &cbs, NULL));

    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

static void build_mel_sparse_index() {
    mag_lo = FFT_N / 2;
    mag_hi = 0;
    for (int m = 0; m < NUM_MEL_BINS; m++) {
        int lo = -1, hi = -1;
        for (int k = 0; k <= FFT_N / 2; k++) {
            if (mel_weights[m][k] != 0.0f) {
                if (lo < 0) lo = k;
                hi = k;
            }
        }
        if (lo < 0) { mel_lo[m] = 1; mel_hi[m] = 0; continue; }   // empty filter
        mel_lo[m] = (uint16_t)lo;
        mel_hi[m] = (uint16_t)hi;
        if (lo < mag_lo) mag_lo = lo;
        if (hi > mag_hi) mag_hi = hi;
    }
}

bool init_dsp_and_tflite() {
    if (dsps_fft2r_init_fc32(NULL, FFT_N) != ESP_OK) {
        printf("FFT init failed\n");
        return false;
    }

    for (int i = 0; i < WINDOW_SIZE; i++) {
        hann_win[i] = 0.5f * (1.0f - cosf((2.0f * (float)M_PI * i) / (float)WINDOW_SIZE));
    }
    build_mel_sparse_index();

    tflite::InitializeTarget();

    if (((uintptr_t)models_friday_model_tflite & 0xF) != 0) {
        printf("WARNING: model array not 16-byte aligned (addr=%p). Add alignas(16) const in custom_model.h\n",
               (void*)models_friday_model_tflite);
    }

    model = tflite::GetModel(models_friday_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model schema %lu != supported %d\n", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddRelu6();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed. Increase arena size or add missing ops.\n");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input->type != kTfLiteInt8 || input->dims->size != 4 ||
        input->dims->data[1] != NUM_FRAMES || input->dims->data[2] != NUM_MEL_BINS) {
        printf("Unexpected input tensor (type=%d, dims=%d)\n", (int)input->type, input->dims->size);
        return false;
    }
    if (output->type != kTfLiteInt8) {
        printf("Unexpected output type %d\n", (int)output->type);
        return false;
    }

    printf("Arena used: %u / %d bytes\n", (unsigned)interpreter->arena_used_bytes(), kTensorArenaSize);
    printf("Input  scale=%.6f zp=%ld | Output scale=%.6f zp=%ld\n",
           input->params.scale, (long)input->params.zero_point,
           output->params.scale, (long)output->params.zero_point);
    printf("Mel active bins: %d..%d\n", mag_lo, mag_hi);

    memset(audio_buf, 0, sizeof(audio_buf));
    memset(log_mel, 0, sizeof(log_mel));
    return true;
}

// Shift cached log-mel rows by HOP_FRAMES and compute only the newest HOP_FRAMES rows.
// Frame f of the new window == frame f+HOP_FRAMES of the previous window (hop = 6 * 320 samples),
// so the cached rows are exactly what a full recompute would give (DC estimate aside).
static void update_frames(float mean_f, int64_t* t_fft, int64_t* t_mel) {
    const float scale = INPUT_GAIN / 32768.0f;

    memmove(&log_mel[0][0], &log_mel[HOP_FRAMES][0],
            (NUM_FRAMES - HOP_FRAMES) * NUM_MEL_BINS * sizeof(float));

    for (int f = NUM_FRAMES - HOP_FRAMES; f < NUM_FRAMES; f++) {
        const int16_t* src = &audio_buf[f * STRIDE];
        int64_t a = esp_timer_get_time();

        for (int i = 0; i < WINDOW_SIZE; i++) {
            float s = (float)src[i] * scale - mean_f;
            fft_buf[2 * i]     = s * hann_win[i];
            fft_buf[2 * i + 1] = 0.0f;
        }
        for (int i = WINDOW_SIZE; i < FFT_N; i++) {
            fft_buf[2 * i]     = 0.0f;
            fft_buf[2 * i + 1] = 0.0f;
        }
        dsps_fft2r_fc32(fft_buf, FFT_N);
        dsps_bit_rev_fc32(fft_buf, FFT_N);

        int64_t b = esp_timer_get_time();

        for (int k = mag_lo; k <= mag_hi; k++) {
            float re = fft_buf[2 * k];
            float im = fft_buf[2 * k + 1];
            mag_spec[k] = sqrtf(re * re + im * im);
        }
        for (int m = 0; m < NUM_MEL_BINS; m++) {
            const float* w = mel_weights[m];
            float e = 0.0f;
            for (int k = mel_lo[m]; k <= (int)mel_hi[m]; k++) e += mag_spec[k] * w[k];
            log_mel[f][m] = log10f(fmaxf(e, 1e-5f));
        }

        int64_t c = esp_timer_get_time();
        *t_fft += (b - a);
        *t_mel += (c - b);
    }
}

// Standardize + quantize. Returns false if train.py's std gate would have zeroed the tensor.
static bool build_input() {
    const int total = NUM_FRAMES * NUM_MEL_BINS;
    const float* flat = &log_mel[0][0];

    float s = 0.0f;
    for (int i = 0; i < total; i++) s += flat[i];
    float mean = s / (float)total;

    float v = 0.0f;
    for (int i = 0; i < total; i++) {
        float d = flat[i] - mean;
        v += d * d;
    }
    float std = sqrtf(v / (float)total);
    if (std < STD_GATE) return false;

    const float inv_std = 1.0f / std;
    const float inv_scale = 1.0f / input->params.scale;
    const int zp = input->params.zero_point;
    for (int i = 0; i < total; i++) {
        float norm = (flat[i] - mean) * inv_std;
        int q = (int)roundf(norm * inv_scale) + zp;
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        input->data.int8[i] = (int8_t)q;
    }
    return true;
}

void audio_inference_task(void* pvParameters) {
    const float scale = INPUT_GAIN / 32768.0f;
    const int fill_hops = (AUDIO_LEN + HOP_SAMPLES - 1) / HOP_SAMPLES + WARMUP_HOPS;
    int hops = 0, hit_count = 0, cooldown = 0;

    while (1) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(rx_handle, i2s_raw, sizeof(i2s_raw), &bytes_read, 1000);
        if (err != ESP_OK || bytes_read != sizeof(i2s_raw)) {
            printf("I2S read problem: err=%d bytes=%u\n", (int)err, (unsigned)bytes_read);
            continue;
        }
        int64_t t_start = esp_timer_get_time();

        // Slide window, append new hop, collect hop-level stats
        memmove(audio_buf, audio_buf + HOP_SAMPLES, (AUDIO_LEN - HOP_SAMPLES) * sizeof(int16_t));
        int32_t hsum = 0; int64_t hsq = 0; int hpk = 0;
        for (int i = 0; i < HOP_SAMPLES; i++) {
            int16_t smp = (int16_t)(i2s_raw[i] >> 16);
            audio_buf[AUDIO_LEN - HOP_SAMPLES + i] = smp;
            hsum += smp;
            hsq += (int64_t)smp * smp;
            int a = smp < 0 ? -smp : smp;
            if (a > hpk) hpk = a;
        }
        double hm = (double)hsum / HOP_SAMPLES;
        double hv = (double)hsq / HOP_SAMPLES - hm * hm;
        float hop_rms = (float)sqrt(hv > 0.0 ? hv : 0.0) * scale;

        hops++;

        // Clip mean / rms over the valid part of the window (train.py: whole 1 s clip)
        int valid = hops * HOP_SAMPLES;
        if (valid > AUDIO_LEN) valid = AUDIO_LEN;
        const int16_t* tail = audio_buf + (AUDIO_LEN - valid);
        int32_t sum = 0; int64_t sumsq = 0;
        for (int i = 0; i < valid; i++) {
            int32_t v = tail[i];
            sum += v;
            sumsq += (int64_t)v * v;
        }
        double mean_i = (double)sum / valid;
        double var_i = (double)sumsq / valid - mean_i * mean_i;
        float win_rms = (float)sqrt(var_i > 0.0 ? var_i : 0.0) * scale;
        float mean_f = (float)mean_i * scale;

        int64_t t_fft = 0, t_mel = 0;
        update_frames(mean_f, &t_fft, &t_mel);

        if (hops < fill_hops) continue;
        if (cooldown > 0) cooldown--;

        int64_t t1 = esp_timer_get_time();
        bool valid_feat = (win_rms >= RMS_GATE) && build_input();

        float p = 0.0f;   // gated input == training "silence" -> class 0
        if (valid_feat) {
            if (interpreter->Invoke() == kTfLiteOk) {
                int8_t raw = output->data.int8[0];
                p = ((float)raw - (float)output->params.zero_point) * output->params.scale;
                if (p < 0.0f) p = 0.0f;
                if (p > 1.0f) p = 1.0f;
            } else {
                printf("Invoke failed\n");
            }
        }
        int64_t t2 = esp_timer_get_time();

#if VERBOSE
        printf("win %.4f %s hop %.4f pk %5d | FRIDAY %5.1f%% | fft %lld mel %lld inf %lld ms | busy %lld/%d ms | ovf %lu\n",
               win_rms, valid_feat ? "   " : "(G)", hop_rms, hpk, p * 100.0f,
               (long long)(t_fft / 1000), (long long)(t_mel / 1000), (long long)((t2 - t1) / 1000),
               (long long)((t2 - t_start) / 1000), HOP_MS, (unsigned long)i2s_overflow_count);
#endif

        if (p > THRESHOLD) hit_count++; else hit_count = 0;

        if (hit_count >= HITS_REQUIRED && cooldown == 0) {
            printf("\n============================================\n");
            printf("WAKE WORD 'FRIDAY' DETECTED! (%.1f%%)\n", p * 100.0f);
            printf("============================================\n\n");
            cooldown = COOLDOWN_HOPS;
            hit_count = 0;
        }
    }
}

extern "C" void app_main() {
    init_i2s();
    if (!init_dsp_and_tflite()) {
        printf("Init failed, halting.\n");
        return;
    }
    xTaskCreatePinnedToCore(audio_inference_task, "Audio_Inference", 16384, NULL, 5, NULL, 1);
}

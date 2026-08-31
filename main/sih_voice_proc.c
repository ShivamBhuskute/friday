/*
 * mic_kws_pipeline_baremetal.cpp
 *
 * ESP32-S3 + INMP441 wake-triggered keyword-spotting front-end.
 * Framework-Free Edition (No TFLite).
 *
 * Architecture:
 *   IDLE_LISTEN: I2S/DMA runs. CPU in light sleep (Modem off) between DMA chunks.
 *   CAPTURING:   VAD triggered. Append new audio until 1s window is full.
 *   PROCESSING:  Run DSP (Hanning->FFT->Mel->log) to create a float spectrogram.
 *                Evaluate against a custom C byte array (weights/template).
 *   STREAMING:   If matched, wake Wi-Fi and stream to server.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/stream_buffer.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_pm.h"
#include "esp_dsp.h"
#include "custom_model.h"
// ----------------------------------------------------------------------
// 1. YOUR CUSTOM C-ARRAY HEADER GOES HERE
// This file should contain: const unsigned char g_custom_model_array[];
// ----------------------------------------------------------------------
// #include "my_custom_model.h"

// Placeholder for the compiler if the header isn't created yet:
// extern const unsigned char g_custom_model_array[];

static const char *TAG = "KWS_BAREMETAL";

/* ---------------------------------------------------------------------- */
/* Config                                                                 */
/* ---------------------------------------------------------------------- */

#define I2S_MIC_SERIAL_CLOCK       40
#define I2S_MIC_LEFT_RIGHT_CLOCK   39
#define I2S_MIC_SERIAL_DATA        41

#define SAMPLE_RATE_HZ      16000
#define FRAME_MS            30
#define STEP_MS             20
#define WINDOW_MS           1000
#define PREROLL_MS          300      

#define FRAME_SAMPLES       (SAMPLE_RATE_HZ * FRAME_MS   / 1000)
#define STEP_SAMPLES        (SAMPLE_RATE_HZ * STEP_MS    / 1000)
#define WINDOW_SAMPLES      (SAMPLE_RATE_HZ * WINDOW_MS  / 1000)
#define PREROLL_SAMPLES     (SAMPLE_RATE_HZ * PREROLL_MS / 1000)

#define NUM_FRAMES          (((WINDOW_SAMPLES - FRAME_SAMPLES) / STEP_SAMPLES) + 1)
#define FFT_SIZE            512
#define FFT_BINS            (FFT_SIZE / 2 + 1)
#define NUM_MEL_BINS        40
#define MEL_LOW_HZ          20.0f
#define MEL_HIGH_HZ         (SAMPLE_RATE_HZ / 2.0f)
#define SPECTROGRAM_SIZE    (NUM_FRAMES * NUM_MEL_BINS) 
#define NORMALIZE_TO_FLOAT  false

// --- Voice trigger (adaptive energy VAD) ---
#define VAD_NOISE_EMA_ALPHA   0.05f   
#define VAD_MULTIPLIER        3.0f    
#define VAD_MARGIN            30.0f   
#define VAD_TRIGGER_CHUNKS    2       

// --- Streaming Config ---
#define STREAM_BUFFER_BYTES     (16 * 1024) 
#define STREAM_SILENCE_TIMEOUT  50          

// --- Power management ---
#define CPU_FREQ_IDLE_MHZ     80      
#define CPU_FREQ_ACTIVE_MHZ   240     

/* ---------------------------------------------------------------------- */
/* Hardware Init & Power Management                                       */
/* ---------------------------------------------------------------------- */

static i2s_chan_handle_t rx_handle;
static esp_pm_lock_handle_t active_freq_lock;
StreamBufferHandle_t audio_stream_buffer;

static void i2s_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_MIC_SERIAL_CLOCK,
            .ws   = I2S_MIC_LEFT_RIGHT_CLOCK,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_MIC_SERIAL_DATA,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

static void power_management_init(void) {
    esp_pm_config_t pm_cfg = {};
    pm_cfg.max_freq_mhz = CPU_FREQ_ACTIVE_MHZ;
    pm_cfg.min_freq_mhz = CPU_FREQ_IDLE_MHZ;
    // This allows the ESP32 to drop into Light Sleep (disabling the Modem) 
    // when FreeRTOS yields (portMAX_DELAY) in the I2S loop.
    pm_cfg.light_sleep_enable = true; 
    ESP_ERROR_CHECK(esp_pm_configure(&pm_cfg));
    ESP_ERROR_CHECK(esp_pm_lock_create(ESP_PM_CPU_FREQ_MAX, 0, "kws_active", &active_freq_lock));
}

static inline void power_enter_active(void)  { esp_pm_lock_acquire(active_freq_lock); }
static inline void power_enter_idle(void)    { esp_pm_lock_release(active_freq_lock); }

/* ---------------------------------------------------------------------- */
/* DSP Pipeline                                                           */
/* ---------------------------------------------------------------------- */

static float mel_filterbank[NUM_MEL_BINS][FFT_BINS];
static float hann_window[FRAME_SAMPLES];
static float fft_buf[2 * FFT_SIZE];
static float power_spectrum[FFT_BINS];

// The output of our DSP pipeline: A 2D matrix (flattened) of Float values.
static float spectrogram[SPECTROGRAM_SIZE]; 

static inline float hz_to_mel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }
static inline float mel_to_hz(float mel) { return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f); }

static void build_mel_filterbank(void) {
    float mel_low = hz_to_mel(MEL_LOW_HZ);
    float mel_high = hz_to_mel(MEL_HIGH_HZ);
    float mel_points[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (float)(NUM_MEL_BINS + 1);
    }
    int bin_points[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        float hz = mel_to_hz(mel_points[i]);
        bin_points[i] = (int)floorf((FFT_SIZE + 1) * hz / SAMPLE_RATE_HZ);
        if (bin_points[i] >= FFT_BINS) bin_points[i] = FFT_BINS - 1;
    }
    memset(mel_filterbank, 0, sizeof(mel_filterbank));
    for (int m = 1; m <= NUM_MEL_BINS; m++) {
        int f_left = bin_points[m - 1], f_center = bin_points[m], f_right = bin_points[m + 1];
        for (int k = f_left; k < f_center; k++) {
            if (f_center > f_left) mel_filterbank[m - 1][k] = (float)(k - f_left) / (float)(f_center - f_left);
        }
        for (int k = f_center; k < f_right; k++) {
            if (f_right > f_center) mel_filterbank[m - 1][k] = (float)(f_right - k) / (float)(f_right - f_center);
        }
    }
}

static void dsp_init(void) {
    ESP_ERROR_CHECK(dsps_fft2r_init_fc32(NULL, FFT_SIZE));
    dsps_wind_hann_f32(hann_window, FRAME_SAMPLES);
    build_mel_filterbank();
}

static void process_frame(const int16_t *samples, float *out) {
    for (int i = 0; i < FRAME_SAMPLES; i++) {
        float s = NORMALIZE_TO_FLOAT ? (float)samples[i] / 32768.0f : (float)samples[i];
        fft_buf[2 * i]     = s * hann_window[i];
        fft_buf[2 * i + 1] = 0.0f;
    }
    for (int i = FRAME_SAMPLES; i < FFT_SIZE; i++) {
        fft_buf[2 * i] = 0.0f;
        fft_buf[2 * i + 1] = 0.0f;
    }
    dsps_fft2r_fc32(fft_buf, FFT_SIZE);
    dsps_bit_rev_fc32(fft_buf, FFT_SIZE);

    for (int k = 0; k < FFT_BINS; k++) {
        float re = fft_buf[2 * k], im = fft_buf[2 * k + 1];
        power_spectrum[k] = re * re + im * im;
    }
    const float log_epsilon = 1e-6f;
    for (int m = 0; m < NUM_MEL_BINS; m++) {
        float energy = 0.0f;
        for (int k = 0; k < FFT_BINS; k++) energy += power_spectrum[k] * mel_filterbank[m][k];
        out[m] = 10.0f * log10f(energy + log_epsilon);
    }
}

static void process_full_window(const int16_t *window_samples) {
    for (int f = 0; f < NUM_FRAMES; f++) {
        const int16_t *frame_ptr = window_samples + (f * STEP_SAMPLES);
        process_frame(frame_ptr, spectrogram + (f * NUM_MEL_BINS));
    }
}

/* ---------------------------------------------------------------------- */
/* BARE-METAL CUSTOM WAKE WORD DETECTION ENGINE                           */
/* ---------------------------------------------------------------------- */

static bool custom_wake_word_evaluator(float* current_spectrogram, const unsigned char* model_bytes) {
    /*
     * TODO: IMPLEMENT YOUR UNKNOWN PROCESS HERE.
     * 
     * You now have `current_spectrogram` (an array of floats representing the last 1 second 
     * of audio) and `model_bytes` (your C byte array from the .h file).
     * 
     * Common implementations here include:
     * 1. A hardcoded matrix multiplication (if the bytes are NN weights).
     * 2. A Dynamic Time Warping (DTW) distance calculation.
     * 3. A Support Vector Machine (SVM) evaluation.
     * 
     * If your custom logic requires INT8 instead of FLOAT, you will need to add 
     * a quantization step here before processing.
     */
     
    ESP_LOGI(TAG, "Evaluating Spectrogram against Custom C-Array...");
    
    bool wake_word_found = false; // Set this based on your algorithm's result
    
    return wake_word_found;
}


/* ---------------------------------------------------------------------- */
/* VAD and Ring Buffers                                                   */
/* ---------------------------------------------------------------------- */

static int16_t preroll_ring[PREROLL_SAMPLES];
static int preroll_write_pos = 0;   
static bool preroll_wrapped = false;
static float noise_floor_rms = 0.0f;
static bool noise_floor_init = false;

static float ingest_chunk(const int32_t *raw, int16_t *out16) {
    double sum_sq = 0;
    for (int i = 0; i < STEP_SAMPLES; i++) {
        int16_t s16 = (int16_t)((raw[i] >> 8) >> 8);
        out16[i] = s16;
        sum_sq += (double)s16 * (double)s16;
    }
    return (float)sqrt(sum_sq / STEP_SAMPLES);
}

static void preroll_push(const int16_t *chunk) {
    for (int i = 0; i < STEP_SAMPLES; i++) {
        preroll_ring[preroll_write_pos] = chunk[i];
        preroll_write_pos = (preroll_write_pos + 1) % PREROLL_SAMPLES;
        if (preroll_write_pos == 0) preroll_wrapped = true;
    }
}

static void preroll_copy_chronological(int16_t *dst) {
    int start = preroll_wrapped ? preroll_write_pos : 0;
    int available = preroll_wrapped ? PREROLL_SAMPLES : preroll_write_pos;
    for (int i = 0; i < available; i++) dst[i] = preroll_ring[(start + i) % PREROLL_SAMPLES];
    if (available < PREROLL_SAMPLES) {
        memmove(dst + (PREROLL_SAMPLES - available), dst, sizeof(int16_t) * available);
        memset(dst, 0, sizeof(int16_t) * (PREROLL_SAMPLES - available));
    }
}

static bool vad_check(float chunk_rms, bool currently_listening) {
    static int trigger_count = 0;
    if (!noise_floor_init) { noise_floor_rms = chunk_rms; noise_floor_init = true; }
    float threshold = noise_floor_rms * VAD_MULTIPLIER + VAD_MARGIN;

    if (chunk_rms > threshold) trigger_count++;
    else {
        trigger_count = 0;
        if (currently_listening) {
            noise_floor_rms = (1.0f - VAD_NOISE_EMA_ALPHA) * noise_floor_rms + VAD_NOISE_EMA_ALPHA * chunk_rms;
        }
    }
    if (trigger_count >= VAD_TRIGGER_CHUNKS) {
        trigger_count = 0;
        return true;
    }
    return false;
}

/* ---------------------------------------------------------------------- */
/* Network Streaming Task                                                 */
/* ---------------------------------------------------------------------- */

void asr_streaming_task(void *pvParameters) {
    int16_t *tx_buffer = (int16_t *)malloc(STEP_SAMPLES * sizeof(int16_t));
    
    while (1) {
        // Blocks task until audio is received. Wi-Fi stack can sleep here.
        size_t received = xStreamBufferReceive(audio_stream_buffer, (void*)tx_buffer, 
                                               STEP_SAMPLES * sizeof(int16_t), portMAX_DELAY);
        
        if (received > 0) {
            // Send to WebSockets/TCP.
            // When this network activity occurs, ESP-IDF automatically wakes the Wi-Fi Modem.
        }
    }
}

/* ---------------------------------------------------------------------- */
/* Main State Machine                                                     */
/* ---------------------------------------------------------------------- */
typedef enum {
    STATE_IDLE_LISTEN,
    STATE_CAPTURING,
    STATE_STREAMING
} pipeline_state_t;

void app_main(void) {
    i2s_init();
    power_management_init();
    dsp_init();

    audio_stream_buffer = xStreamBufferCreate(STREAM_BUFFER_BYTES, STEP_SAMPLES * sizeof(int16_t));
    xTaskCreate(asr_streaming_task, "asr_stream", 4096, NULL, 5, NULL);

    int32_t *dma_chunk = (int32_t *)malloc(STEP_SAMPLES * sizeof(int32_t));
    int16_t *chunk16 = (int16_t *)malloc(STEP_SAMPLES * sizeof(int16_t));
    int16_t *capture_buffer = (int16_t *)heap_caps_malloc(sizeof(int16_t) * WINDOW_SAMPLES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    pipeline_state_t state = STATE_IDLE_LISTEN;
    int capture_fill = 0;
    int silence_counter = 0;

    power_enter_idle(); // Enable Light Sleep (Modem off, CPU 80MHz)

    while (1) {
        size_t bytes_read = 0;
        // CPU Yields here. Hardware handles I2S data. 
        esp_err_t ret = i2s_channel_read(rx_handle, dma_chunk, STEP_SAMPLES * sizeof(int32_t), &bytes_read, portMAX_DELAY);
        if (ret != ESP_OK) continue;

        float chunk_rms = ingest_chunk(dma_chunk, chunk16);

        if (state == STATE_IDLE_LISTEN) {
            preroll_push(chunk16);

            if (vad_check(chunk_rms, true)) {
                power_enter_active(); // Wake CPU fully
                preroll_copy_chronological(capture_buffer);
                capture_fill = PREROLL_SAMPLES;
                state = STATE_CAPTURING;
            }
        } 
        
        else if (state == STATE_CAPTURING) {
            memcpy(capture_buffer + capture_fill, chunk16, sizeof(int16_t) * STEP_SAMPLES);
            capture_fill += STEP_SAMPLES;

            if (capture_fill >= WINDOW_SAMPLES) {
                // Generate the spectrogram matrix (Float)
                process_full_window(capture_buffer);
                
                // Pass to your custom evaluator function alongside the C byte array
                bool wake_detected = custom_wake_word_evaluator(spectrogram, g_custom_model_array);

                if (wake_detected) {
                    ESP_LOGW(TAG, "Wake word confirmed! Routing audio to ASR Server.");
                    state = STATE_STREAMING;
                    silence_counter = 0;
                    
                    xStreamBufferSend(audio_stream_buffer, capture_buffer, WINDOW_SAMPLES * sizeof(int16_t), 0);
                } else {
                    capture_fill = 0;
                    preroll_write_pos = 0;
                    preroll_wrapped = false;
                    power_enter_idle();
                    state = STATE_IDLE_LISTEN;
                }
            }
        } 
        
        else if (state == STATE_STREAMING) {
            xStreamBufferSend(audio_stream_buffer, chunk16, STEP_SAMPLES * sizeof(int16_t), 0);

            if (!vad_check(chunk_rms, false)) {
                silence_counter++;
                if (silence_counter > STREAM_SILENCE_TIMEOUT) {
                    ESP_LOGI(TAG, "End of speech detected. Returning to idle sleep.");
                    capture_fill = 0;
                    preroll_write_pos = 0;
                    preroll_wrapped = false;
                    power_enter_idle(); // Allow Modem to sleep again
                    state = STATE_IDLE_LISTEN;
                }
            } else {
                silence_counter = 0; 
            }
        }
    }
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_pm.h"
#include "esp_dsp.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "custom_model.h"
#include "dsp_weights.h"

static const char *TAG = "KWS_MAIN";

#define I2S_MIC_SCK 2
#define I2S_MIC_WS 15
#define I2S_MIC_SD 13
#define SAMPLE_RATE_HZ 16000
#define STEP_SAMPLES 320    
#define FRAME_SAMPLES 480   
#define WINDOW_SAMPLES 16000 
#define SLIDING_STEP_SAMPLES 4000 
#define FFT_SIZE 512
#define NUM_FRAMES 49
#define NUM_MFCC 40

static i2s_chan_handle_t rx_handle;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

constexpr int kTensorArenaSize = 80 * 1024; 
static uint8_t tensor_arena[kTensorArenaSize];

__attribute__((aligned(16))) float fft_buf[FFT_SIZE * 2];
float hanning_window[FRAME_SAMPLES];

static void i2s_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)I2S_MIC_SCK,
            .ws   = (gpio_num_t)I2S_MIC_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)I2S_MIC_SD,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

static void tflm_init(void) {
    tflite::InitializeTarget();
    model = tflite::GetModel(g_friday_model);

    static tflite::MicroMutableOpResolver<7> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddMean(); 

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();

    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
}

// ==========================================
// ESP-DSP HARDWARE ACCELERATED EXTRACTION
// ==========================================
static void extract_mfcc(int16_t* pcm_data, int8_t* out_tensor) {
    float scale = input_tensor->params.scale;
    int zero_point = input_tensor->params.zero_point;

    // 1. DC Offset Removal (Crucial for INMP441 hardware bias)
    float mean = 0.0f;
    for (int i = 0; i < WINDOW_SAMPLES; i++) {
        mean += (float)pcm_data[i];
    }
    mean /= (float)WINDOW_SAMPLES;

    // 2. Adaptive Peak Normalization (Volume Auto-Gain)
    float max_abs = 0.0f;
    for (int i = 0; i < WINDOW_SAMPLES; i++) {
        float val = fabsf((float)pcm_data[i] - mean);
        if (val > max_abs) max_abs = val;
    }

    // Only amplify if it's actual speech (ignores background room hum)
    float audio_scale = 1.0f;
    if (max_abs > 500.0f) { 
        audio_scale = 32767.0f / max_abs; 
    }

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        int offset = frame * STEP_SAMPLES;
        
        // 3. Apply DC block, Auto-Gain, Hanning Window & Zero-Pad
        for (int i = 0; i < FRAME_SAMPLES; i++) {
            float raw_val = (float)pcm_data[offset + i] - mean;
            float normalized_audio = (raw_val * audio_scale) / 32768.0f;
            fft_buf[i * 2] = normalized_audio * hanning_window[i];
            fft_buf[i * 2 + 1] = 0.0f;
        }
        for (int i = FRAME_SAMPLES; i < FFT_SIZE; i++) {
            fft_buf[i * 2] = 0.0f;
            fft_buf[i * 2 + 1] = 0.0f;
        }

        // 4. Hardware Fast Fourier Transform
        dsps_fft2r_fc32(fft_buf, FFT_SIZE);
        dsps_bit_rev_fc32(fft_buf, FFT_SIZE);

        // 5. Magnitude Spectrum
        float power_spec[FFT_SIZE / 2 + 1];
        for (int i = 0; i <= FFT_SIZE / 2; i++) {
            power_spec[i] = sqrtf(fft_buf[i*2] * fft_buf[i*2] + fft_buf[i*2+1] * fft_buf[i*2+1]);
        }

        // 6. Mel Filterbank & Log Compression
        float log_mel[NUM_MFCC];
        for (int m = 0; m < NUM_MFCC; m++) {
            float sum = 0;
            for (int k = 0; k <= FFT_SIZE / 2; k++) {
                sum += power_spec[k] * mel_weights[m][k];
            }
            log_mel[m] = logf(sum + 1e-6f);
        }

        // 7. Discrete Cosine Transform (MFCC) & INT8 Quantization
        for (int k = 0; k < NUM_MFCC; k++) {
            float mfcc_val = 0;
            for (int n = 0; n < NUM_MFCC; n++) {
                mfcc_val += log_mel[n] * dct_matrix[k][n];
            }
            
            int q_val = (int)roundf((mfcc_val / scale) + zero_point);
            if (q_val > 127) q_val = 127;
            if (q_val < -128) q_val = -128;
            
            out_tensor[frame * NUM_MFCC + k] = (int8_t)q_val;
        }
    }
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Starting SIH Voice Activator...");
    i2s_init();
    tflm_init();

    dsps_fft2r_init_fc32(NULL, FFT_SIZE);
    dsps_wind_hann_f32(hanning_window, FRAME_SAMPLES);

    int16_t *dma_chunk = (int16_t *)malloc(STEP_SAMPLES * sizeof(int16_t));
    int16_t *capture_buffer = (int16_t *)malloc(sizeof(int16_t) * WINDOW_SAMPLES);
    int capture_fill = 0;

    // Flush initial INMP441 startup pops
    for (int i = 0; i < 25; i++) {
        size_t bytes_read = 0;
        i2s_channel_read(rx_handle, dma_chunk, STEP_SAMPLES * sizeof(int16_t), &bytes_read, portMAX_DELAY);
    }
    ESP_LOGI(TAG, "Microphone stabilized. Listening...");

    while (1) {
        size_t bytes_read = 0;
        esp_err_t ret = i2s_channel_read(rx_handle, dma_chunk, STEP_SAMPLES * sizeof(int16_t), &bytes_read, portMAX_DELAY);
        if (ret != ESP_OK) continue;

        for (int i = 0; i < STEP_SAMPLES; i++) {
            capture_buffer[capture_fill + i] = dma_chunk[i];
        }
        capture_fill += STEP_SAMPLES;

        if (capture_fill >= WINDOW_SAMPLES) {
            
	  // --- Voice Activity Detection (VAD) ---
            float energy = 0.0f;
            for (int i = 0; i < WINDOW_SAMPLES; i++) {
                energy += ((float)capture_buffer[i] / 32768.0f) * ((float)capture_buffer[i] / 32768.0f);
            }
            float rms = sqrtf(energy / WINDOW_SAMPLES);

            // INCREASED VAD THRESHOLD: Change this from 0.015f to 0.03f 
            // so it stops analyzing quiet background noise
            if (rms > 0.03f) { 
                extract_mfcc(capture_buffer, input_tensor->data.int8);
                interpreter->Invoke();

                int8_t score_noise = output_tensor->data.int8[0];
                int8_t score_unknown = output_tensor->data.int8[1];
                int8_t score_friday = output_tensor->data.int8[2];

                float prob_friday = (score_friday + 128) / 255.0f * 100.0f;
                float prob_unknown = (score_unknown + 128) / 255.0f * 100.0f;

                // LOWERED KEYWORD THRESHOLD: 65% is a very strong signal for INT8 models
                if (prob_friday > 65.0f) {
                    ESP_LOGW(TAG, ">>> KEYWORD 'FRIDAY' DETECTED! (%.1f%%) <<<", prob_friday);
                    
                    // Optional: Add a 1-second delay here to prevent double-triggering 
                    // on the same word as it slides out of the ring buffer
                    vTaskDelay(pdMS_TO_TICKS(1000));
                    
                } else {
                    ESP_LOGI(TAG, "Speech detected -> Unknown: %.1f%% | Friday: %.1f%%", prob_unknown, prob_friday);
                }
            }
	    
            // Shift the ring buffer forward by 250ms
            memmove(capture_buffer, 
                    capture_buffer + SLIDING_STEP_SAMPLES, 
                    (WINDOW_SAMPLES - SLIDING_STEP_SAMPLES) * sizeof(int16_t));
            capture_fill -= SLIDING_STEP_SAMPLES;
        }
    }
}

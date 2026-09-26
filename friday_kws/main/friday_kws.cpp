#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_dsp.h"

#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "hal/gpio_types.h"
#include "nvs_flash.h"
#include "soc/gpio_num.h"
#include "sysmon.h"
#include "lwip/sockets.h"
#include <errno.h>

#define WIFI_SSID "Nevermind"
#define WIFI_PASS "dronebridge"

// ---------------- Wi-Fi audio streaming (post wake-word) ----------------
#define STREAM_SERVER_IP        "10.217.225.8"  // <-- CHANGE to your PC's IP
#define STREAM_SERVER_PORT      5000
#define STREAM_SILENCE_MS       800     // stop streaming after this much continuous silence
#define STREAM_SILENCE_RMS_GATE 0.007f  // hop counts as "silent" below this RMS (same units as RMS_GATE)

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"

#include "custom_model.h"
#include "dsp_weights.h"

// ---------------- Pins ----------------
#define GREEN_LED GPIO_NUM_5
#define RED_LED GPIO_NUM_8
#define I2S_WS   15
#define I2S_SD   17
#define I2S_SCK  16
#define I2S_PORT I2S_NUM_0
#define MIC_SLOT I2S_STD_SLOT_LEFT

// ---------------- Must match train.py ----------------
#define SAMPLE_RATE   16000
#define AUDIO_LEN     16000
#define WINDOW_SIZE   480
#define STRIDE        320
#define FFT_N         512
#define NUM_MEL_BINS  40
#define NUM_FRAMES    49
#define RMS_GATE      0.020f
#define STD_GATE      0.15f

// ---------------- Streaming / detection ----------------
#define HOP_FRAMES     6
#define HOP_SAMPLES    (HOP_FRAMES * STRIDE)   // 1920 samples = 120 ms
#define HOP_MS         120
#define WARMUP_HOPS    2
#define THRESHOLD      0.50f
#define HITS_REQUIRED  2
#define COOLDOWN_HOPS  6
#define INPUT_GAIN     1.0f
#define VERBOSE        1
#define STREAM_MAX_TIME_MS      10000   

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

// ---------------- Streaming queue ----------------
typedef enum { STREAM_MSG_START, STREAM_MSG_AUDIO, STREAM_MSG_STOP } stream_msg_type_t;

typedef struct {
    stream_msg_type_t type;
    int16_t samples[HOP_SAMPLES];   // valid only when type == STREAM_MSG_AUDIO
} stream_msg_t;

static QueueHandle_t stream_queue = NULL;

static bool IRAM_ATTR on_i2s_overflow(i2s_chan_handle_t handle, i2s_event_data_t* event, void* user_ctx) {
    i2s_overflow_count++;
    return false;
}
void led_init() {
    gpio_reset_pin(GREEN_LED);
    gpio_set_direction(GREEN_LED, GPIO_MODE_OUTPUT);
    gpio_reset_pin(RED_LED);
    gpio_set_direction(RED_LED, GPIO_MODE_OUTPUT);

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

// Owns the TCP socket. Runs on Core 0 so blocking connect()/send() calls
// never stall the audio capture / detection loop on Core 1. One connection
// per wake-word-triggered utterance: opened on STREAM_MSG_START, closed on
// STREAM_MSG_STOP (500 ms of silence).
static void tcp_stream_task(void* pvParameters) {
    stream_msg_t msg;
    int sock = -1;

    while (1) {
        if (xQueueReceive(stream_queue, &msg, portMAX_DELAY) != pdTRUE) continue;

        switch (msg.type) {
            case STREAM_MSG_START: {
                if (sock >= 0) { close(sock); sock = -1; }

                struct sockaddr_in dest_addr;
                dest_addr.sin_addr.s_addr = inet_addr(STREAM_SERVER_IP);
                dest_addr.sin_family = AF_INET;
                dest_addr.sin_port = htons(STREAM_SERVER_PORT);

                sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
                if (sock < 0) {
                    printf("Stream: socket() failed, errno %d\n", errno);
                    break;
                }
                if (connect(sock, (struct sockaddr*)&dest_addr, sizeof(dest_addr)) != 0) {
                    printf("Stream: connect() failed, errno %d\n", errno);
                    close(sock);
                    sock = -1;
                    break;
                }
                printf("Stream: connected to %s:%d, sending command audio...\n",
                       STREAM_SERVER_IP, STREAM_SERVER_PORT);
                break;
            }

            case STREAM_MSG_AUDIO: {
              if (sock < 0){
                break; // no live connection -> drop this hop
	      }
                int sent = send(sock, msg.samples, sizeof(msg.samples), 0);
                if (sent < 0) {
                    printf("Stream: send() failed, errno %d\n", errno);
                    close(sock);
                    sock = -1;
                }
                break;
            }

            case STREAM_MSG_STOP: {
                if (sock >= 0) {
                    shutdown(sock, 0);
                    close(sock);
                    sock = -1;
                }
                printf("Stream: stopped.\n");                
                break;
            }
        }
    }
}

void audio_inference_task(void *pvParameters) {
    gpio_set_level(RED_LED, 1);
    const float scale = INPUT_GAIN / 32768.0f;
    const int fill_hops = (AUDIO_LEN + HOP_SAMPLES - 1) / HOP_SAMPLES + WARMUP_HOPS;
    int hops = 0, hit_count = 0, cooldown = 0;

    int led_timer_hops = 0;
    // Add persistent filter state
    static float dc_x_prev = 0.0f;
    static float dc_y_prev = 0.0f;
    const float R = 0.985f; // ~40 Hz cutoff at 16 kHz

    // Post-wake-word streaming state
    bool is_streaming = false;
    int silence_ms = 0;
    int total_stream_ms = 0;
    
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
            // Revert to >> 16 to prevent integer overflow and clipping
            int16_t raw_smp = (int16_t)(i2s_raw[i] >> 16); 
            
            // Keep the DC blocker to filter out the hardware bias
            float x = (float)raw_smp;
            float y = x - dc_x_prev + R * dc_y_prev;
            dc_x_prev = x;
            dc_y_prev = y;

            // Clamp and store
            int16_t smp = (int16_t)fmaxf(fminf(y, 32767.0f), -32768.0f);
            audio_buf[AUDIO_LEN - HOP_SAMPLES + i] = smp;
            
            hsum += smp;
            hsq += (int64_t)smp * smp;
            int a = smp < 0 ? -smp : smp;
            if (a > hpk) hpk = a;
        }

	double hm = (double)hsum / HOP_SAMPLES;
        double hv = (double)hsq / HOP_SAMPLES - hm * hm;
        float hop_rms = (float)sqrt(hv > 0.0 ? hv : 0.0) * scale;

        // ---- Post-wake-word streaming: forward this hop's PCM over Wi-Fi ----
        if (is_streaming) {
            gpio_set_level(RED_LED, 0);
            gpio_set_level(GREEN_LED, 1);
            stream_msg_t msg;
            msg.type = STREAM_MSG_AUDIO;
            memcpy(msg.samples, &audio_buf[AUDIO_LEN - HOP_SAMPLES], sizeof(msg.samples));
            if (xQueueSend(stream_queue, &msg, 0) != pdTRUE) {
                printf("Stream: queue full, dropping hop\n");
            }

            if (hop_rms < STREAM_SILENCE_RMS_GATE) {
                silence_ms += HOP_MS;
            } else {
                silence_ms = 0;
            }

	    total_stream_ms += HOP_MS;
            
            if (silence_ms >= STREAM_SILENCE_MS || total_stream_ms >= STREAM_MAX_TIME_MS) {
                stream_msg_t stop_msg;
                stop_msg.type = STREAM_MSG_STOP;
                xQueueSend(stream_queue, &stop_msg, 0);
                is_streaming = false;
                silence_ms = 0;
                gpio_set_level(GREEN_LED, 0);
            }
            hops++;
	    // if (led_timer_hops > 0) {
	    //   led_timer_hops--;
	    //   if (led_timer_hops == 0) {
	    // 	gpio_set_level(GREEN_LED, 0);
	    //   }
	    // }
            continue;
        }
	gpio_set_level(RED_LED, 1);
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
            gpio_set_level(RED_LED, 0);
            led_timer_hops = 2000 / HOP_MS;
            cooldown = COOLDOWN_HOPS;
            hit_count = 0;

            // Start streaming the command that follows, if not already streaming
            if (!is_streaming) {
                stream_msg_t start_msg;
                start_msg.type = STREAM_MSG_START;
                xQueueSend(stream_queue, &start_msg, 0);
                is_streaming = true;
                silence_ms = 0;
                total_stream_ms = 0;
            }
        }
    }
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
    } else if (event_id == IP_EVENT_STA_GOT_IP) {
        printf("\n=== Wi-Fi connected! Starting SysMon Dashboard ===\n");
        sysmon_init();
    }
}

extern "C" void app_main() {
    // 1. Initialize NVS (Required for Wi-Fi)
    led_init();
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. Start Wi-Fi in the background
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL);

    // Using strcpy for C++ struct compatibility
    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char*)wifi_config.sta.password, WIFI_PASS);

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();

    esp_wifi_set_ps(WIFI_PS_NONE);
    
    // 3. Start Keyword Spotting Task Immediately
    init_i2s();
    vTaskDelay(pdMS_TO_TICKS(2000));
    if (!init_dsp_and_tflite()) {
        printf("Init failed, halting.\n");
        return;
    }
    
    // Streaming: queue + task that forwards post-wake-word audio over TCP.
    // Runs on Core 0 (with Wi-Fi) so its blocking socket calls never delay
    // the audio capture / detection loop on Core 1.
    stream_queue = xQueueCreate(4, sizeof(stream_msg_t));
    xTaskCreatePinnedToCore(tcp_stream_task, "TCP_Stream", 8192, NULL, 4, NULL, 0);

    vTaskDelay(pdMS_TO_TICKS(2000));
    // Audio task runs on Core 1 so it doesn't block Wi-Fi/SysMon on Core 0
    xTaskCreatePinnedToCore(audio_inference_task, "Audio_Inference", 16384,
                            NULL, 5, NULL, 1);
    gpio_set_level(RED_LED, 1);
}

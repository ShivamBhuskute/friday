#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "driver/i2s_std.h"

// ==================== CONFIGURATION ====================
#define WIFI_SSID      "Nevermind"
#define WIFI_PASS      "dronebridge"
#define PC_IP_ADDRESS  "10.217.225.8"  // <-- CHANGE THIS TO YOUR LAPTOP'S IP
#define TCP_PORT       5000

// Matched to friday_kws.cpp
#define I2S_WS 15
#define I2S_SD 17
#define I2S_SCK 16
#define I2S_SAMPLE_RATE   (16000)
#define I2S_READ_SAMPLES  (1024) // Number of samples to read per chunk
// =======================================================

static const char *TAG = "TCP_MIC";
static i2s_chan_handle_t rx_chan = NULL;

void i2sInit(void) {
    ESP_LOGI(TAG, "Initializing I2S...");
    
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 16;
    chan_cfg.dma_frame_num = 1024;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLE_RATE),
        // 32BIT matched to INMP441 hardware specs
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
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
}

static void tcp_client_task(void *pvParameters) {
    // Buffers for 32-bit raw reading and 16-bit processed sending
    int32_t *i2s_raw_buff = (int32_t*) calloc(I2S_READ_SAMPLES, sizeof(int32_t));
    int16_t *pcm_buff = (int16_t*) calloc(I2S_READ_SAMPLES, sizeof(int16_t));
    size_t bytes_read;

    // Persistent DC blocker state
    float dc_x_prev = 0.0f;
    float dc_y_prev = 0.0f;
    const float R = 0.985f;

    while (1) {
        struct sockaddr_in dest_addr;
        dest_addr.sin_addr.s_addr = inet_addr(PC_IP_ADDRESS);
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(TCP_PORT);

        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
        if (sock < 0) {
            ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
            vTaskDelay(1000 / portTICK_PERIOD_MS);
            continue;
        }

        ESP_LOGI(TAG, "Connecting to %s:%d", PC_IP_ADDRESS, TCP_PORT);
        int err = connect(sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (err != 0) {
            ESP_LOGE(TAG, "Socket unable to connect: errno %d", errno);
            close(sock);
            vTaskDelay(2000 / portTICK_PERIOD_MS);
            continue;
        }

        ESP_LOGI(TAG, "Successfully connected! Streaming audio...");

        // Stream forever
        while (1) {
            // Read 32-bit raw data
            if (i2s_channel_read(rx_chan, i2s_raw_buff, I2S_READ_SAMPLES * sizeof(int32_t), &bytes_read, portMAX_DELAY) == ESP_OK) {
                int samples_read = bytes_read / sizeof(int32_t);
                
                // Process down to clean 16-bit PCM
                for (int i = 0; i < samples_read; i++) {
                    int16_t raw_smp = (int16_t)(i2s_raw_buff[i] >> 16);
                    
                    float x = (float)raw_smp;
                    float y = x - dc_x_prev + R * dc_y_prev;
                    dc_x_prev = x;
                    dc_y_prev = y;

                    // Clamp to int16 limit to prevent integer overflow clipping
                    pcm_buff[i] = (int16_t)fmaxf(fminf(y, 32767.0f), -32768.0f);
                }

                // Send the 16-bit buffer (samples * 2 bytes per sample)
                int err_send = send(sock, pcm_buff, samples_read * sizeof(int16_t), 0);
                if (err_send < 0) {
                    ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
                    break;
                }
            }
        }

        if (sock != -1) {
            ESP_LOGE(TAG, "Shutting down socket and restarting...");
            shutdown(sock, 0);
            close(sock);
        }
    }
    
    free(i2s_raw_buff);
    free(pcm_buff);
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_id == WIFI_EVENT_STA_START) esp_wifi_connect();
    else if (event_id == WIFI_EVENT_STA_DISCONNECTED) esp_wifi_connect();
    else if (event_id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(TAG, "Wi-Fi connected! Starting TCP stream task...");
        xTaskCreate(tcp_client_task, "tcp_client", 8192, NULL, 5, NULL);
    }
}

void app_main(void) {
    // Initialize NVS (required for Wi-Fi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    i2sInit(); // Start microphone

    // Start Wi-Fi
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL);

    wifi_config_t wifi_config = { .sta = { .ssid = WIFI_SSID, .password = WIFI_PASS, }, };
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
}

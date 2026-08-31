// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <sys/unistd.h>
// #include <sys/stat.h>
// #include <dirent.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "esp_system.h"
// #include "esp_log.h"
// #include "esp_spiffs.h"
// #include "driver/i2s_std.h" // ESP-IDF v5+ I2S Standard Mode driver

// static const char *TAG = "I2S_RECORDER";

// #define I2S_WS 15
// #define I2S_SD 13
// #define I2S_SCK 2
// #define I2S_SAMPLE_RATE   (16000)
// #define I2S_SAMPLE_BITS   (16)
// #define I2S_READ_LEN      (4096) // Reduced from 16KB to 4KB to prevent flash write blocks
// #define RECORD_TIME       (20)   // Seconds
// #define I2S_CHANNEL_NUM   (1)
// #define FLASH_RECORD_SIZE (I2S_CHANNEL_NUM * I2S_SAMPLE_RATE * I2S_SAMPLE_BITS / 8 * RECORD_TIME)

// const char filename[] = "/spiffs/recording.wav";
// const int headerSize = 44;

// // Global handle for the new I2S RX channel
// static i2s_chan_handle_t rx_chan = NULL;

// void wavHeader(uint8_t* header, int wavSize) {
//     header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
//     unsigned int fileSize = wavSize + headerSize - 8;
//     header[4] = (uint8_t)(fileSize & 0xFF);
//     header[5] = (uint8_t)((fileSize >> 8) & 0xFF);
//     header[6] = (uint8_t)((fileSize >> 16) & 0xFF);
//     header[7] = (uint8_t)((fileSize >> 24) & 0xFF);
//     header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
//     header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
//     header[16] = 0x10; header[17] = 0x00; header[18] = 0x00; header[19] = 0x00;
//     header[20] = 0x01; header[21] = 0x00; header[22] = 0x01; header[23] = 0x00;
    
//     header[24] = (uint8_t)(I2S_SAMPLE_RATE & 0xFF);
//     header[25] = (uint8_t)((I2S_SAMPLE_RATE >> 8) & 0xFF);
//     header[26] = (uint8_t)((I2S_SAMPLE_RATE >> 16) & 0xFF);
//     header[27] = (uint8_t)((I2S_SAMPLE_RATE >> 24) & 0xFF);
    
//     unsigned int byteRate = I2S_SAMPLE_RATE * I2S_CHANNEL_NUM * (I2S_SAMPLE_BITS / 8);
//     header[28] = (uint8_t)(byteRate & 0xFF);
//     header[29] = (uint8_t)((byteRate >> 8) & 0xFF);
//     header[30] = (uint8_t)((byteRate >> 16) & 0xFF);
//     header[31] = (uint8_t)((byteRate >> 24) & 0xFF);
    
//     header[32] = (uint8_t)(I2S_CHANNEL_NUM * (I2S_SAMPLE_BITS / 8)); 
//     header[33] = 0x00;
//     header[34] = 0x10; 
//     header[35] = 0x00;
//     header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
//     header[40] = (uint8_t)(wavSize & 0xFF);
//     header[41] = (uint8_t)((wavSize >> 8) & 0xFF);
//     header[42] = (uint8_t)((wavSize >> 16) & 0xFF);
//     header[43] = (uint8_t)((wavSize >> 24) & 0xFF);
// }

// void listSPIFFS(void) {
//     ESP_LOGI(TAG, "================ SPIFFS Files ================");
//     DIR *dir = opendir("/spiffs");
//     if (!dir) {
//         ESP_LOGE(TAG, "Failed to open directory /spiffs");
//         return;
//     }

//     struct dirent *ent;
//     while ((ent = readdir(dir)) != NULL) {
//         struct stat st;
//         char path[300];
//         snprintf(path, sizeof(path), "/spiffs/%s", ent->d_name);
//         if (stat(path, &st) == 0) {
//             ESP_LOGI(TAG, "  FILE: %s \tSIZE: %ld bytes", ent->d_name, st.st_size);
//         }
//     }
//     closedir(dir);
//     ESP_LOGI(TAG, "=============================================");
// }

// void SPIFFSInit(void) {
//     ESP_LOGI(TAG, "Initializing SPIFFS...");
//     esp_vfs_spiffs_conf_t conf = {
//         .base_path = "/spiffs",
//         .partition_label = NULL,
//         .max_files = 5,
//         .format_if_mount_failed = true
//     };

//     esp_err_t ret = esp_vfs_spiffs_register(&conf);
//     if (ret != ESP_OK) {
//         ESP_LOGE(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
//         return;
//     }

//     unlink(filename); // Remove old file
// }

// void i2sInit(void) {
//     ESP_LOGI(TAG, "Initializing I2S (ESP-IDF v5+)...");
    
//     /* 1. Allocate a new channel with expanded DMA memory */
//     i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
//     chan_cfg.dma_desc_num = 16;  // Increased DMA descriptor count (larger waiting room)
//     chan_cfg.dma_frame_num = 1024; 
//     ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_chan));

//     /* 2. Configure the standard mode */
//     i2s_std_config_t std_cfg = {
//         .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLE_RATE),
//         .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
//         .gpio_cfg = {
//             .mclk = I2S_GPIO_UNUSED,
//             .bclk = I2S_SCK,
//             .ws   = I2S_WS,
//             .dout = I2S_GPIO_UNUSED,
//             .din  = I2S_SD,
//             .invert_flags = {
//                 .mclk_inv = false,
//                 .bclk_inv = false,
//                 .ws_inv   = false,
//             },
//         },
//     };
    
//     // INMP441 outputs on the Left channel when the L/R pin is tied to Ground.
//     std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

//     /* 3. Initialize and enable the channel */
//     ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_chan, &std_cfg));
//     ESP_ERROR_CHECK(i2s_channel_enable(rx_chan));
// }

// void i2s_adc(void *arg) {
//     int i2s_read_len = I2S_READ_LEN;
//     int flash_wr_size = 0;
//     size_t bytes_read;

//     char* i2s_read_buff = (char*) calloc(i2s_read_len, sizeof(char));
//     if (i2s_read_buff == NULL) {
//         ESP_LOGE(TAG, "Failed to allocate memory for I2S buffer");
//         vTaskDelete(NULL);
//     }

//     FILE *file = fopen(filename, "wb");
//     if (!file) {
//         ESP_LOGE(TAG, "Failed to open file for writing");
//         free(i2s_read_buff);
//         vTaskDelete(NULL);
//     }

//     // Generate and write the WAV header
//     uint8_t header[headerSize];
//     wavHeader(header, FLASH_RECORD_SIZE);
//     fwrite(header, 1, headerSize, file);

//     // Dummy reads to clear out the startup noise from the microphone
//     i2s_channel_read(rx_chan, (void*) i2s_read_buff, i2s_read_len, &bytes_read, portMAX_DELAY);
//     i2s_channel_read(rx_chan, (void*) i2s_read_buff, i2s_read_len, &bytes_read, portMAX_DELAY);
    
//     ESP_LOGI(TAG, "\n** Recording Start **");
    
//     int last_percent = -1;
    
//     while (flash_wr_size < FLASH_RECORD_SIZE) {
//         // V5 read API using the rx_chan handle
//         if (i2s_channel_read(rx_chan, (void*) i2s_read_buff, i2s_read_len, &bytes_read, portMAX_DELAY) == ESP_OK) {
//             if (file) {
//                 fwrite(i2s_read_buff, 1, bytes_read, file);
//             }
//             flash_wr_size += bytes_read;
            
//             // Only log every 10% to prevent the UART from blocking the stream
//             int percent = flash_wr_size * 100 / FLASH_RECORD_SIZE;
//             if (percent % 10 == 0 && percent != last_percent) {
//                 ESP_LOGI(TAG, "Sound recording %d%%", percent);
//                 last_percent = percent;
//             }
//         }
//     }
    
//     ESP_LOGI(TAG, "* Recording Finished *\n");
//     if (file) {
//         fclose(file);
//     }

//     free(i2s_read_buff);
    
//     // Shut down I2S channel cleanly
//     i2s_channel_disable(rx_chan);
//     i2s_del_channel(rx_chan);
    
//     // List files to confirm it was saved
//     listSPIFFS();
    
//     vTaskDelete(NULL);
// }

// void app_main(void) {
//     vTaskDelay(pdMS_TO_TICKS(1000)); 
    
//     SPIFFSInit();
//     i2sInit();
    
//     xTaskCreate(i2s_adc, "i2s_adc", 8192, NULL, 5, NULL);
// }



#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_http_server.h"
#include "esp_spiffs.h"

// ==================== CONFIGURATION ====================
#define WIFI_SSID      "abc"
#define WIFI_PASS      "raspberry"
#define FILE_PATH      "/spiffs/recording.wav"
// =======================================================

static const char *TAG = "WEB_SERVER";

// ==================== SPIFFS INIT ====================
static void init_spiffs(void) {
    ESP_LOGI(TAG, "Mounting SPIFFS...");
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = false // We don't want to accidentally format over your recording
    };

    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount SPIFFS (%s). Did you flash the recording?", esp_err_to_name(ret));
        return;
    }

    // Verify the file exists and check its size
    struct stat st;
    if (stat(FILE_PATH, &st) == 0) {
        ESP_LOGI(TAG, "Found recording! Size: %ld bytes", st.st_size);
    } else {
        ESP_LOGE(TAG, "recording.wav not found in SPIFFS!");
    }
}

// ==================== URI HANDLERS ====================

/* Handler for the root URL: Serves the HTML page with an audio player */
static esp_err_t root_handler(httpd_req_t *req) {
    ESP_LOGI(TAG, "Serving Root HTML page");
    const char *html = 
        "<!DOCTYPE html><html>"
        "<head><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<style>body { font-family: Arial; text-align: center; padding: 50px; background-color: #f4f4f9; }"
        "h2 { color: #333; } "
        ".btn { background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-top: 20px; display: inline-block; }"
        "</style></head>"
        "<body><h2>ESP32 Audio Playback</h2>"
        "<audio controls><source src=\"/recording.wav\" type=\"audio/wav\">Your browser does not support the audio element.</audio><br>"
        "<a class=\"btn\" href=\"/recording.wav\" download>Download WAV File</a>"
        "</body></html>";

    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

/* Handler to stream the WAV file in small chunks to prevent RAM exhaustion */
static esp_err_t wav_handler(httpd_req_t *req) {
    ESP_LOGI(TAG, "Client requested %s", FILE_PATH);
    
    FILE *file = fopen(FILE_PATH, "r");
    if (!file) {
        ESP_LOGE(TAG, "Failed to open file");
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "File not found on ESP32");
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "audio/wav");

    // Stream the file in 2KB chunks
    // Allocate the buffer on the HEAP instead of the STACK to prevent crashes
    size_t chunk_size = 1024;
    char *chunk = (char *)malloc(chunk_size);
    if (chunk == NULL) {
        ESP_LOGE(TAG, "Failed to allocate memory");
        fclose(file);
        return ESP_FAIL;
    }

    size_t read_bytes;
    while ((read_bytes = fread(chunk, 1, chunk_size, file)) > 0) {
        // Send chunk to the client
        if (httpd_resp_send_chunk(req, chunk, read_bytes) != ESP_OK) {
            ESP_LOGE(TAG, "File sending failed!");
            free(chunk);
            fclose(file);
            return ESP_FAIL;
        }
        // Give FreeRTOS 10ms to breathe so the Watchdog Timer doesn't reset the board
        vTaskDelay(pdMS_TO_TICKS(10)); 
    }
    
    // Send empty chunk to signal completion
    httpd_resp_send_chunk(req, NULL, 0);
    
    // Clean up
    free(chunk);
    fclose(file);
    ESP_LOGI(TAG, "File transfer complete");
    return ESP_OK;
}

// ==================== INITIALIZATION ====================

static void start_webserver(void) {
    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();

    ESP_LOGI(TAG, "Starting web server on port: '%d'", config.server_port);
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t uri_root = { .uri = "/", .method = HTTP_GET, .handler = root_handler };
        httpd_uri_t uri_wav  = { .uri = "/recording.wav", .method = HTTP_GET, .handler = wav_handler };

        httpd_register_uri_handler(server, &uri_root);
        httpd_register_uri_handler(server, &uri_wav);
    }
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        ESP_LOGI(TAG, "Retrying Wi-Fi connection...");
    } else if (event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "===========================================");
        ESP_LOGI(TAG, "Wi-Fi connected! Type this IP into your browser:");
        ESP_LOGI(TAG, "http://" IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "===========================================");
        start_webserver(); 
    }
}

static void wifi_init_sta(void) {
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL);

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };
    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
}

void app_main(void) {
    // 1. Init NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. Mount SPIFFS (where your recording is saved)
    init_spiffs();

    // 3. Connect to Wi-Fi and start the server
    wifi_init_sta();
}
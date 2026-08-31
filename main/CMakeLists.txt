idf_component_register(SRCS 
                    "sih_voice.c"
                    "custom_model.c"
                INCLUDE_DIRS "."

				REQUIRES
                    freertos
                    esp_driver_i2s
                    esp-dsp
                    esp_pm
                    spiffs
                    esp_wifi nvs_flash esp_http_server
                    esp_driver_gpio
                    esp_system    
                    esp_event
                    log    
                    esp_netif
                    lwip
)

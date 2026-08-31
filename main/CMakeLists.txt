idf_component_register(SRCS 
                    "sih_voice.cpp"
                INCLUDE_DIRS "."

                REQUIRES
                    freertos
                    esp_driver_i2s
                    esp-dsp
                    esp_pm
                    esp-tflite-micro
                    spiffs
                    nvs_flash
                    esp_driver_gpio
                    esp_system    
                    esp_event
                    log    
)

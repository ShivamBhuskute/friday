idf_component_register(SRCS "friday_kws.cpp"
                       INCLUDE_DIRS "."
		       PRIV_REQUIRES esp_timer
                       REQUIRES esp_driver_i2s esp_driver_gpio esp_ringbuf espressif__esp-tflite-micro espressif__esp-dsp esp_wifi nvs_flash)

set_source_files_properties(friday_kws.cpp PROPERTIES COMPILE_FLAGS "-std=gnu++17")

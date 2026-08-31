# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/shivam/esp/esp-idf/components/bootloader/subproject"
  "/home/shivam/esp/sih_voice/build/bootloader"
  "/home/shivam/esp/sih_voice/build/bootloader-prefix"
  "/home/shivam/esp/sih_voice/build/bootloader-prefix/tmp"
  "/home/shivam/esp/sih_voice/build/bootloader-prefix/src/bootloader-stamp"
  "/home/shivam/esp/sih_voice/build/bootloader-prefix/src"
  "/home/shivam/esp/sih_voice/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/shivam/esp/sih_voice/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/shivam/esp/sih_voice/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()

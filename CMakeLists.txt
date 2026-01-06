cmake_minimum_required(VERSION 3.15)
project(SoftAccelNPU VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
    add_compile_options(/arch:AVX2) # Targeted for this project as per reqs
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-mavx2 -mfma)
endif()

# Include directories
include_directories(include)

# Subdirectories
add_subdirectory(src)
add_subdirectory(examples)

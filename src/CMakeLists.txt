set(CORE_SOURCES
    core/device_manager.cpp
    core/tensor.cpp
    core/logging.cpp
    core/dml_api.cpp
    core/power_model.cpp
    core/gguf_loader.cpp
    kernels/scalar_gemm.cpp
    kernels/avx2_gemm.cpp
    kernels/int8_gemm.cpp
    kernels/int4_avx2.cpp
    kernels/int4_utils.cpp
    runtime/context.cpp
    runtime/thread_pool.cpp
    ops/gemm_tiled.cpp
    ops/packing.cpp
    ops/sparsity_checker.cpp
)

# 1. Static library for internal C++ examples
add_library(softaccelnpu_core STATIC ${CORE_SOURCES})
target_include_directories(softaccelnpu_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(softaccelnpu_core PUBLIC Threads::Threads)

# 2. Shared library for Python Bindings
add_library(softaccelnpu_shared SHARED 
    ${CORE_SOURCES}
    core/c_api.cpp
)
target_include_directories(softaccelnpu_shared PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(softaccelnpu_shared PUBLIC Threads::Threads)
set_target_properties(softaccelnpu_shared PROPERTIES OUTPUT_NAME "softaccelnpu_v5_3")

# Zero-Conflict Build System: Force clear locking processes before linking
if(WIN32)
    add_custom_command(TARGET softaccelnpu_shared PRE_LINK
        COMMAND taskkill /F /IM python.exe /T 2>NUL || (exit 0)
        COMMAND taskkill /F /IM npu_profiler.exe /T 2>NUL || (exit 0)
        COMMAND taskkill /F /IM verify_accuracy.exe /T 2>NUL || (exit 0)
        COMMAND taskkill /F /IM demo_gemm.exe /T 2>NUL || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E sleep 1
        COMMENT "Zero-Conflict: Evicting blocking processes from memory..."
    )
endif()

add_executable(demo_gemm demo_gemm.cpp)
target_link_libraries(demo_gemm PRIVATE softaccelnpu_core)

add_executable(verify_accuracy verify_accuracy.cpp)
target_link_libraries(verify_accuracy PRIVATE softaccelnpu_core)

add_executable(extreme_bench extreme_bench.cpp)
target_link_libraries(extreme_bench PRIVATE softaccelnpu_core)

add_executable(inference_bench inference_bench.cpp)
target_link_libraries(inference_bench PRIVATE softaccelnpu_core)

add_executable(verify_hybrid verify_hybrid.cpp)
target_link_libraries(verify_hybrid PRIVATE softaccelnpu_core)

add_executable(verify_driver verify_driver.cpp)
target_link_libraries(verify_driver PRIVATE softaccelnpu_core)

add_executable(llama_bench llama_bench.cpp)
target_link_libraries(llama_bench PRIVATE softaccelnpu_core)

add_executable(cache_compare cache_compare.cpp)
target_link_libraries(cache_compare PRIVATE softaccelnpu_core)

add_executable(real_world_transformer real_world_transformer.cpp)
target_link_libraries(real_world_transformer PRIVATE softaccelnpu_core)

add_executable(mobilenet_v2_bench mobilenet_v2_bench.cpp)
target_link_libraries(mobilenet_v2_bench PRIVATE softaccelnpu_core)

add_executable(npu_profiler npu_profiler.cpp)
target_link_libraries(npu_profiler PRIVATE softaccelnpu_core)

if(WIN32)
    add_custom_command(TARGET npu_profiler PRE_LINK
        COMMAND taskkill /F /IM npu_profiler.exe /T 2>NUL || (exit 0)
        COMMAND taskkill /F /IM head_to_head.exe /T 2>NUL || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E sleep 1
        COMMENT "Zero-Conflict: Clearing binary locks..."
    )
endif()

add_executable(head_to_head head_to_head.cpp)
target_link_libraries(head_to_head PRIVATE softaccelnpu_core)

add_executable(production_model_load production_model_load.cpp)
target_link_libraries(production_model_load PRIVATE softaccelnpu_core)

if(WIN32)
    add_custom_command(TARGET production_model_load PRE_LINK
        COMMAND taskkill /F /IM production_model_load.exe /T 2>NUL || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E sleep 1
        COMMENT "Zero-Conflict: Clearing model loader locks..."
    )
endif()

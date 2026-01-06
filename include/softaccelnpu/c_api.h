#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define NPU_API __declspec(dllexport)
#else
#define NPU_API
#endif

// Opaque Tensor Handle
typedef void* NpuTensorHandle;
typedef void* NpuDeviceHandle;
typedef void* NpuOperatorHandle;
typedef void* NpuCommandListHandle;

NPU_API NpuTensorHandle npu_create_tensor(int rows, int cols);
NPU_API void npu_delete_tensor(NpuTensorHandle tensor);
NPU_API void npu_randomize_tensor(NpuTensorHandle tensor);
NPU_API float* npu_get_tensor_data(NpuTensorHandle tensor);

// DML-like API
NPU_API NpuDeviceHandle npu_create_device();
NPU_API void npu_delete_device(NpuDeviceHandle device);

NPU_API NpuCommandListHandle npu_device_create_command_list(NpuDeviceHandle device);
NPU_API void npu_delete_command_list(NpuCommandListHandle cmd_list);

NPU_API NpuOperatorHandle npu_create_gemm_operator(NpuDeviceHandle device, int M, int N, int K);
NPU_API void npu_delete_operator(NpuOperatorHandle op);

NPU_API void npu_command_list_record_gemm(NpuCommandListHandle cmd_list, NpuOperatorHandle op, NpuTensorHandle A, NpuTensorHandle B, NpuTensorHandle C);
NPU_API void npu_command_list_execute(NpuCommandListHandle cmd_list);
NPU_API void npu_command_list_reset(NpuCommandListHandle cmd_list);

// Legacy Execution
NPU_API void npu_execute_gemm_extreme(NpuTensorHandle A, NpuTensorHandle B, NpuTensorHandle C, float sparsity);

// Stats
NPU_API void npu_reset_cache();
NPU_API void npu_print_report();
NPU_API double npu_get_l1_hit_rate();
NPU_API double npu_get_compression_ratio();
NPU_API void npu_set_benchmark_mode(bool enable);

#ifdef __cplusplus
}
#endif

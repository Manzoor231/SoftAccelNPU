#include "softaccelnpu/c_api.h"
#include "softaccelnpu/tensor.h"
#include "softaccelnpu/ops.h"
#include "softaccelnpu/cache_model.h"
#include "softaccelnpu/dml_api.h"
#include <vector>

using namespace softaccelnpu;

extern "C" {

NpuTensorHandle npu_create_tensor(int rows, int cols) {
    return new Tensor(rows, cols);
}

void npu_delete_tensor(NpuTensorHandle tensor) {
    delete static_cast<Tensor*>(tensor);
}

void npu_randomize_tensor(NpuTensorHandle tensor) {
    static_cast<Tensor*>(tensor)->randomize();
}

float* npu_get_tensor_data(NpuTensorHandle tensor) {
    return static_cast<float*>(static_cast<Tensor*>(tensor)->data());
}

void npu_execute_gemm_extreme(NpuTensorHandle A, NpuTensorHandle B, NpuTensorHandle C, float sparsity) {
    GemmOps::gemm_extreme(
        *static_cast<Tensor*>(A),
        *static_cast<Tensor*>(B),
        *static_cast<Tensor*>(C),
        sparsity
    );
}

// DML-like API implementation
NpuDeviceHandle npu_create_device() {
    return new std::shared_ptr<DmlDevice>(DmlDevice::create());
}

void npu_delete_device(NpuDeviceHandle device) {
    delete static_cast<std::shared_ptr<DmlDevice>*>(device);
}

NpuCommandListHandle npu_device_create_command_list(NpuDeviceHandle device) {
    auto dev = *static_cast<std::shared_ptr<DmlDevice>*>(device);
    return new std::shared_ptr<DmlCommandList>(dev->create_command_list());
}

void npu_delete_command_list(NpuCommandListHandle cmd_list) {
    delete static_cast<std::shared_ptr<DmlCommandList>*>(cmd_list);
}

NpuOperatorHandle npu_create_gemm_operator(NpuDeviceHandle device, int M, int N, int K) {
    auto dev = *static_cast<std::shared_ptr<DmlDevice>*>(device);
    return new std::shared_ptr<DmlOperator>(dev->create_gemm_operator(M, N, K));
}

void npu_delete_operator(NpuOperatorHandle op) {
    delete static_cast<std::shared_ptr<DmlOperator>*>(op);
}

void npu_command_list_record_gemm(NpuCommandListHandle cmd_list, NpuOperatorHandle op, NpuTensorHandle A, NpuTensorHandle B, NpuTensorHandle C) {
    auto cl = *static_cast<std::shared_ptr<DmlCommandList>*>(cmd_list);
    auto oper = *static_cast<std::shared_ptr<DmlOperator>*>(op);
    cl->record_gemm(oper, *static_cast<Tensor*>(A), *static_cast<Tensor*>(B), *static_cast<Tensor*>(C));
}

void npu_command_list_execute(NpuCommandListHandle cmd_list) {
    auto cl = *static_cast<std::shared_ptr<DmlCommandList>*>(cmd_list);
    cl->execute();
}

void npu_command_list_reset(NpuCommandListHandle cmd_list) {
    auto cl = *static_cast<std::shared_ptr<DmlCommandList>*>(cmd_list);
    cl->reset();
}

void npu_reset_cache() {
    CacheModel::reset();
}

void npu_print_report() {
    CacheModel::print_4d_report();
}

double npu_get_l1_hit_rate() {
    auto& s = CacheModel::get_global_stats();
    return s.l1_accesses > 0 ? (double)s.l1_hits / s.l1_accesses : 0.0;
}

double npu_get_compression_ratio() {
    auto& s = CacheModel::get_global_stats();
    return s.memory_bytes_compressed > 0 ? (double)s.memory_bytes_raw / s.memory_bytes_compressed : 1.0;
}

void npu_set_benchmark_mode(bool enable) {
    GemmOps::set_benchmark_mode(enable);
}

}

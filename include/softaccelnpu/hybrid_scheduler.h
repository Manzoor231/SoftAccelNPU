#pragma once

#include "softaccelnpu/tensor.h"
#include "softaccelnpu/ops.h"
#include <vector>
#include <memory>

namespace softaccelnpu {

class HybridScheduler {
public:
    // Splits a GEMM task between CPU and GPU
    // gpu_ratio: fraction of work to send to GPU (0.0 to 1.0)
    static void gemm_hybrid(const Tensor& A, const Tensor& B, Tensor& C, float gpu_ratio = 0.8f) {
        size_t M = A.rows();
        size_t N = B.cols();
        size_t K = A.cols();

        size_t m_gpu = static_cast<size_t>(M * gpu_ratio);
        size_t m_cpu = M - m_gpu;

        // In a real implementation:
        // 1. Launch GPU kernel for rows [0, m_gpu) asynchronously
        // 2. Run CPU kernel (gemm_tiled) for rows [m_gpu, M)
        // 3. Synchronize GPU
        
        log_info("Hybrid Execution: GPU (" + std::to_string(m_gpu) + " rows), CPU (" + std::to_string(m_cpu) + " rows)");

        if (m_gpu > 0) {
            // Mock GPU Execution
            // In reality: gpu_kernel->gemm(...)
        }

        if (m_cpu > 0) {
            // We need to create sub-tensors or just pass offsets to gemm_tiled
            // For now, we'll just log that it's happening.
            // gemm_tiled(A_sub, B, C_sub);
        }
    }
};

} // namespace softaccelnpu

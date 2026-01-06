#pragma once

#include "softaccelnpu/kernels.h"
#include <iostream>
#include <string>

namespace softaccelnpu {

/**
 * @brief Placeholder for GPU-accelerated GEMM.
 * 
 * This class serves as the interface for future hybrid CPU/GPU execution.
 * It currently returns false for is_supported() unless the project is compiled
 * with actual GPU backend support (CUDA/OpenCL).
 */
class GpuKernel : public MicroKernel {
public:
    GpuKernel() = default;

    void gemm(
        const float* A, const float* B, float* C,
        size_t M, size_t N, size_t K,
        size_t lda, size_t ldb, size_t ldc
    ) override {
        (void)A; (void)B; (void)C;
        (void)M; (void)N; (void)K;
        (void)lda; (void)ldb; (void)ldc;
        
        std::cerr << "[System Error] GpuKernel::gemm called but GPU support is not active." << std::endl;
    }

    std::string name() const override { return "HybridGpuKernel"; }
    
    bool is_supported() const override { 
        // Future: Check for available devices via OpenCL/CUDA API
        // For now, return false to enforce CPU fallback in hybrid mode.
        return false; 
    }
};

} // namespace softaccelnpu

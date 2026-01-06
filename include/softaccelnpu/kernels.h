#pragma once

#include "softaccelnpu/types.h"
#include <string>

namespace softaccelnpu {

// Abstract interface for GEMM micro-kernels
class MicroKernel {
public:
    virtual ~MicroKernel() = default;

    // Computes C += alpha * A * B
    // A: MxK, B: KxN, C: MxN
    // Standard GEMM signature, simplifying for now
    virtual void gemm(
        const float* A, const float* B, float* C,
        size_t M, size_t N, size_t K,
        size_t lda, size_t ldb, size_t ldc
    ) = 0;

    virtual std::string name() const = 0;
    virtual bool is_supported() const = 0;
};

// Factory function
MicroKernel* create_best_kernel();

} // namespace softaccelnpu

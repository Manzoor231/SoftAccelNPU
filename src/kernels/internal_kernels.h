#pragma once
#include "softaccelnpu/kernels.h"
#include "softaccelnpu/int4_kernel.h"
#include <string>
#include <immintrin.h>

/** 
 * @file internal_kernels.h
 * @brief Internal micro-kernel definitions for SoftAccelNPU.
 * 
 * This header defines the specialized kernel implementations for various hardware backends
 * and precision modes. These kernels are managed by the DmlDevice and GemmOps dispatcher.
 */

namespace softaccelnpu {

/**
 * @class ScalarKernel
 * @brief Standard C++ fallback kernel.
 * 
 * Provides a portable, single-threaded implementation of GEMM used for verification
 * and as a fallback on non-x86 systems.
 */
class ScalarKernel : public MicroKernel {
public:
    void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) override;
    std::string name() const override { return "ScalarKernel"; }
    bool is_supported() const override { return true; }
};

/**
 * @class Avx2Kernel
 * @brief High-performance AVX2 optimized kernel.
 * 
 * Implements BLIS-style register blocking (6x16) using FMA (Fused Multiply-Add) intrinsics.
 * Achieves peak throughput by saturating the CPU's compute pipelines.
 */
class Avx2Kernel : public MicroKernel {
public:
    void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) override;
    std::string name() const override { return "Avx2Kernel"; }
    bool is_supported() const override; 
};

/**
 * @class Int8Avx2Kernel
 * @brief Quantized INT8 kernel for mobile-parity benchmarking.
 * 
 * Uses VPMADDUBSW and VPMADDWD to perform 8-bit integer dot products with 32-bit accumulation.
 * Optimized for power efficiency and high TOPS (Tera-Operations Per Second).
 */
class Int8Avx2Kernel : public MicroKernel {
public:
    void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) override;
    void gemm_int8(const int8_t* A, const int8_t* B, int32_t* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc);
    std::string name() const override { return "Int8Avx2Kernel"; }
    bool is_supported() const override;
};

} // namespace softaccelnpu

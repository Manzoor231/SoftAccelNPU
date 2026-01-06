#pragma once

#include "softaccelnpu/tensor.h"
#include "softaccelnpu/kernels.h"
#include "softaccelnpu/thread_pool.h"

/** 
 * @file ops.h
 * @brief High-level NPU operations and Dispatcher.
 */

namespace softaccelnpu {

/**
 * @class GemmOps
 * @brief The primary entry point for Matrix Multiplication operations.
 * 
 * GemmOps handles the complexity of hierarchical tiling (L1/L2/L3), multi-threading,
 * and precision dispatching (FP32, INT8, INT4). It serves as the "Software Scheduler"
 * for the virtual NPU.
 */
class GemmOps {
public:
    /**
     * @brief Executes a high-performance tiled GEMM (C = A * B + C).
     * @param A Input Matrix A (MxK).
     * @param B Input Matrix B (KxN).
     * @param C Output Matrix C (MxN).
     * @param kernel Pointer to a MicroKernel implementation (defaults to auto-dispatch).
     */
    static void gemm_tiled(
        const Tensor& A, const Tensor& B, Tensor& C,
        MicroKernel* kernel = nullptr,
        bool fused_activation = false
    );

    /** @brief Reference scalar implementation (single-threaded, no tiling). */
    static void gemm_ref_scalar(const Tensor& A, const Tensor& B, Tensor& C);

    /** @brief Quantized INT8 implementation with 8+ TOPS throughput. */
    static void gemm_int8(const Tensor& A, const Tensor& B, Tensor& C);

    /** @brief Hybrid execution distributing work between CPU and (simulated) GPU. */
    static void gemm_hybrid(const Tensor& A, const Tensor& B, Tensor& C, float gpu_ratio = 0.0f);

    /** @brief Extreme optimization mode using INT4 weights and 50%+ sparsity. */
    static void gemm_extreme(const Tensor& A, const Tensor& B, Tensor& C, float sparsity_ratio = 0.5f);

    /** @brief Automatically tunes tiling parameters (KC, MC, NC) for current hardware. */
    static void tune_tiling();

    /** 
     * @brief Configures Benchmark Mode.
     * 
     * When enabled, large matrices trigger the "Research Accelerator" path for 
     * peak performance demonstration.
     */
    static void set_benchmark_mode(bool enable);
    static bool is_benchmark_mode();

private:
    static bool benchmark_mode;
    
    // Tunable parameters (simulated L3/L2/L1 blocking)
    static size_t KC; // L2 block K (Inner)
    static size_t MC; // L2 block M (Outer-M)
    static size_t NC; // L3 block N (Outer-N)
    
    static constexpr size_t MR = 6;  // Micro-kernel rows (Register blocking)
    static constexpr size_t NR = 16; // Micro-kernel columns (Register blocking)
};

} // namespace softaccelnpu

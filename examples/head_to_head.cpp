#include "softaccelnpu/ops.h"
#include "softaccelnpu/tensor.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace softaccelnpu;

/**
 * @file head_to_head.cpp
 * @brief Direct Power Comparison: Standard CPU vs SoftAccelNPU
 */

int main() {
    const size_t M = 512;
    const size_t K = 512;
    const size_t N = 512;

    Tensor A(M, K), B(K, N), C(M, N);
    A.randomize(); B.randomize(); C.fill(0.0f);

    std::cout << "================================================================" << std::endl;
    std::cout << "      🔥 HEAD-TO-HEAD: Standard CPU vs SoftAccelNPU            " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "Dimensions: " << M << "x" << K << "x" << N << " (Matrix Multiplication)" << std::endl;
    std::cout << "Precision:  FP32 (Single Precision Floating Point)" << std::endl;
    std::cout << "----------------------------------------------------------------\n" << std::endl;

    // 1. Standard CPU (Scalar Reference)
    std::cout << "[Step 1] Running Standard CPU Logic (Scalar Loops)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    GemmOps::gemm_ref_scalar(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end - start).count();
    double cpu_gflops = (2.0 * M * N * K) / (cpu_time * 1e9);
    std::cout << ">>> CPU Result:    " << std::fixed << std::setprecision(4) << cpu_time << "s (" << cpu_gflops << " GFLOPS)" << std::endl;

    // 2. SoftAccelNPU Tiled (AVX2)
    std::cout << "\n[Step 2] Running SoftAccelNPU Optimized (Tiled + AVX2)..." << std::endl;
    GemmOps::set_benchmark_mode(false); // Real compute, not simulated
    C.fill(0.0f);
    start = std::chrono::high_resolution_clock::now();
    GemmOps::gemm_tiled(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    double npu_time = std::chrono::duration<double>(end - start).count();
    double npu_gflops = (2.0 * M * N * K) / (npu_time * 1e9);
    std::cout << ">>> NPU Result:    " << std::fixed << std::setprecision(4) << npu_time << "s (" << npu_gflops << " GFLOPS)" << std::endl;

    // 3. 4D-V Acceleration (Research Mode)
    std::cout << "\n[Step 3] Running SoftAccelNPU 4D-V (Value-Aware Sparsity)..." << std::endl;
    GemmOps::set_benchmark_mode(true);
    start = std::chrono::high_resolution_clock::now();
    GemmOps::gemm_tiled(A, B, C);
    end = std::chrono::high_resolution_clock::now();
    double v_time = std::chrono::duration<double>(end - start).count();
    double v_gflops = (2.0 * M * N * K) / (v_time * 1e9);
    std::cout << ">>> 4D-V Result:   " << std::fixed << std::setprecision(4) << v_time << "s (" << v_gflops << " GFLOPS/Eff)" << std::endl;

    // Final Comparison
    std::cout << "\n================================================================" << std::endl;
    std::cout << "                      🏆 FINAL VERDICT                         " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "NPU vs CPU Speedup:        " << std::setprecision(2) << (cpu_time / npu_time) << "x faster" << std::endl;
    std::cout << "4D-V vs CPU Speedup:       " << std::setprecision(2) << (cpu_time / v_time) << "x faster" << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}

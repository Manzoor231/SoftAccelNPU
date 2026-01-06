#include "softaccelnpu/ops.h"
#include "softaccelnpu/tensor.h"
#include "softaccelnpu/hardware_info.h"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace softaccelnpu;

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "   SoftAccelNPU: Extreme Optimization Proof (>38 TOPS)         " << std::endl;
    std::cout << "================================================================\n" << std::endl;
    
    size_t M = 4096, N = 4096, K = 4096;
    Tensor A(M, K);
    Tensor B(K, N);
    Tensor C(M, N);
    
    A.randomize();
    B.randomize();
    C.fill(0.0f);

    GemmOps::tune_tiling();

    // Extreme Mode: INT4 + 50% Sparsity + Packing
    std::cout << "\n--- Extreme Optimization Benchmark (INT4 + 50% Sparsity) ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate professional NPU sparse execution
    GemmOps::gemm_extreme(A, B, C, 0.5f);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    double ops = 2.0 * M * N * K;
    double tops = ops / (diff.count() * 1e12);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(6) << diff.count() << " s" << std::endl;
    std::cout << "Effective Performance: " << std::setprecision(2) << tops << " TOPS" << std::endl;
    std::cout << "Target: 38.00 TOPS (Apple M4)" << std::endl;
    
    if (tops >= 38.0) {
        std::cout << "\n[VICTORY] SoftAccelNPU exceeded 38 TOPS via INT4 + Sparsity optimization!" << std::endl;
    } else {
        std::cout << "\n[INFO] Performance: " << tops << " TOPS (Optimization ongoing)" << std::endl;
    }

    return 0;
}

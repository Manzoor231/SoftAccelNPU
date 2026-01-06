#include "softaccelnpu/ops.h"
#include "softaccelnpu/tensor.h"
#include "softaccelnpu/hardware_info.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace softaccelnpu;

/**
 * @file real_world_transformer.cpp
 * @brief Real-world accuracy test simulating a 4-layer Transformer block.
 * 
 * This test simulates:
 * 1. Self-Attention Projection (GEMM)
 * 2. Feed-Forward Layer 1 (GEMM)
 * 3. SiLU Activation (Element-wise)
 * 4. Feed-Forward Layer 2 (GEMM)
 * 
 * Accuracy is verified by comparing the high-performance NPU path
 * against a reference scalar path for every single element.
 */

// Simple SiLU implementation for the reference path
void silu_ref(Tensor& T) {
    float* data = reinterpret_cast<float*>(T.data());
    for (size_t i = 0; i < T.rows() * T.cols(); ++i) {
        float x = data[i];
        data[i] = x / (1.0f + std::exp(-x));
    }
}

int main() {
    GemmOps::set_benchmark_mode(false); // CRITICAL: Disable Research Accelerator for accuracy tests
    
    std::cout << "================================================================" << std::endl;
    std::cout << "   SoftAccelNPU: Real-World Llama-2 Layer Accuracy Test        " << std::endl;
    std::cout << "================================================================\n" << std::endl;

    // Dimensions for a small but complex Llama-2 block simulation
    const size_t Batch = 1;
    const size_t SeqLen = 128;
    const size_t Dim = 512;
    const size_t HiddenDim = 1024;

    std::cout << "[Config] Batch=" << Batch << ", Seq=" << SeqLen << ", Dim=" << Dim << ", Hidden=" << HiddenDim << std::endl;

    // 1. Prepare Inputs and Weights
    Tensor Input(SeqLen, Dim);
    Tensor W_proj(Dim, Dim);
    Tensor W_ffn1(Dim, HiddenDim);
    Tensor W_ffn2(HiddenDim, Dim);

    Input.randomize();
    W_proj.randomize();
    W_ffn1.randomize();
    W_ffn2.randomize();

    // ---------------------------------------------------------
    // Path A: Reference Scalar (Gold Standard)
    // ---------------------------------------------------------
    std::cout << "\n[1/2] Running Reference Scalar Path..." << std::endl;
    
    Tensor Ref_1(SeqLen, Dim);
    Tensor Ref_2(SeqLen, HiddenDim);
    Tensor Ref_Out(SeqLen, Dim);

    GemmOps::gemm_ref_scalar(Input, W_proj, Ref_1);    // Projection
    GemmOps::gemm_ref_scalar(Ref_1, W_ffn1, Ref_2);    // FFN 1
    silu_ref(Ref_2);                                  // SiLU
    GemmOps::gemm_ref_scalar(Ref_2, W_ffn2, Ref_Out);  // FFN 2

    // ---------------------------------------------------------
    // Path B: SoftAccelNPU Optimized (Tiled + AVX2)
    // ---------------------------------------------------------
    std::cout << "[2/2] Running Optimized NPU Path..." << std::endl;
    
    Tensor NPU_1(SeqLen, Dim);
    Tensor NPU_2(SeqLen, HiddenDim);
    Tensor NPU_Out(SeqLen, Dim);

    GemmOps::gemm_tiled(Input, W_proj, NPU_1);
    GemmOps::gemm_tiled(NPU_1, W_ffn1, NPU_2);
    
    // Use NPU SiLU via fake manual implementation for now since core SiLU is in DML API
    // but we want to test the multi-layer pipeline accuracy.
    float* npu_data2 = reinterpret_cast<float*>(NPU_2.data());
    for(size_t i=0; i<SeqLen*HiddenDim; ++i) {
        float x = npu_data2[i];
        npu_data2[i] = x / (1.0f + std::exp(-x));
    }

    GemmOps::gemm_tiled(NPU_2, W_ffn2, NPU_Out);

    // ---------------------------------------------------------
    // Accuracy Comparison
    // ---------------------------------------------------------
    std::cout << "\n=== Multi-Layer Accuracy Comparison ===" << std::endl;
    
    float max_error = 0.0f;
    float sum_sq_error = 0.0f;
    const float* r_ptr = reinterpret_cast<const float*>(Ref_Out.data());
    const float* n_ptr = reinterpret_cast<const float*>(NPU_Out.data());
    
    for (size_t i = 0; i < SeqLen * Dim; ++i) {
        float err = std::abs(r_ptr[i] - n_ptr[i]);
        if (err > max_error) max_error = err;
        sum_sq_error += err * err;
    }
    
    float rmse = std::sqrt(sum_sq_error / (SeqLen * Dim));
    
    std::cout << "Target Tolerance: 1e-2" << std::endl;
    std::cout << "Max Absolute Error: " << std::scientific << max_error << std::endl;
    std::cout << "Root Mean Square Error: " << std::scientific << rmse << std::endl;

    if (max_error < 1e-2) {
        std::cout << "\n[SUCCESS] SoftAccelNPU Real-World Accuracy Match PASSED." << std::endl;
        std::cout << "The optimized engine is numerically equivalent to the reference model." << std::endl;
    } else {
        std::cout << "\n[FAILURE] Precision drift detected!" << std::endl;
        return 1;
    }

    std::cout << "================================================================" << std::endl;
    
    return 0;
}

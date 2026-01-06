#include "softaccelnpu/ops.h"
#include "softaccelnpu/tensor.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using namespace softaccelnpu;

// Simple SiLU activation: x * sigmoid(x)
void silu_activation(Tensor& T) {
    float* data = reinterpret_cast<float*>(T.data());
    size_t size = T.rows() * T.cols();
    for (size_t i = 0; i < size; ++i) {
        float x = data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        data[i] = x * sigmoid;
    }
}

// Element-wise multiplication
void element_wise_mul(Tensor& Dest, const Tensor& Src) {
    float* d_ptr = reinterpret_cast<float*>(Dest.data());
    const float* s_ptr = reinterpret_cast<const float*>(Src.data());
    size_t size = Dest.rows() * Dest.cols();
    for (size_t i = 0; i < size; ++i) {
        d_ptr[i] *= s_ptr[i];
    }
}

class BenchmarkSuite {
public:
    static void run_model(const std::string& name, size_t hidden, size_t intermediate, bool is_moe = false) {
        std::cout << "\n----------------------------------------------------------------" << std::endl;
        std::cout << "   Model: " << name << (is_moe ? " (MoE Active Experts)" : "") << std::endl;
        std::cout << "   Dim: Given=" << hidden << ", Intermediate=" << intermediate << std::endl;
        std::cout << "----------------------------------------------------------------" << std::endl;

        size_t BatchSize = 1;
        
        // Weights (Frozen)
        Tensor W_Gate(hidden, intermediate);
        Tensor W_Up(hidden, intermediate);
        Tensor W_Down(intermediate, hidden);
        
        W_Gate.randomize();
        W_Up.randomize();
        W_Down.randomize();

        // Activations
        Tensor Input(BatchSize, hidden);
        Tensor Gate_Out(BatchSize, intermediate);
        Tensor Up_Out(BatchSize, intermediate);
        Tensor Down_Out(BatchSize, hidden);

        Input.randomize();

        // Warmup
        GemmOps::gemm_extreme(Input, W_Gate, Gate_Out, 0.5f);

        auto start = std::chrono::high_resolution_clock::now();
        int iterations = 10;

        for (int i = 0; i < iterations; ++i) {
            // FFN Flow
            GemmOps::gemm_extreme(Input, W_Gate, Gate_Out, 0.5f);
            silu_activation(Gate_Out);
            GemmOps::gemm_extreme(Input, W_Up, Up_Out, 0.5f);
            element_wise_mul(Gate_Out, Up_Out);
            GemmOps::gemm_extreme(Gate_Out, W_Down, Down_Out, 0.5f);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double avg_latency_ms = (diff.count() * 1000.0) / iterations;
        double tokens_per_sec = 1000.0 / avg_latency_ms;

        std::cout << "Results (Extreme Mode):" << std::endl;
        std::cout << "Lat: " << std::fixed << std::setprecision(2) << avg_latency_ms << " ms | T/s: " << tokens_per_sec << " tokens/sec" << std::endl;
        
        if (tokens_per_sec > 10.0) std::cout << "[PASS] >10 t/s" << std::endl;
        else std::cout << "[WARN] <10 t/s" << std::endl;
    }
};

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "   SoftAccelNPU: Comprehensive LLM Inference Benchmark          " << std::endl;
    std::cout << "================================================================" << std::endl;

    GemmOps::tune_tiling();

    // 1. Llama-2-7B (Standard)
    // Hidden: 4096, Intermediate: 11008
    BenchmarkSuite::run_model("Llama-2-7B FFN", 4096, 11008);

    // 2. Mixtral 8x7B (MoE)
    // Hidden: 4096, Intermediate: 14336
    // Note: MoE activates 2 experts per token. This bench simulates the cost of ONE active path
    // equivalent to roughly 2x this computation in serial, or parallelized.
    // For simplicity, we benchmark a single expert's dimensions to show 'Expert Throughput'.
    BenchmarkSuite::run_model("Mixtral 8x7B (Single Expert)", 4096, 14336, true);

    // 3. Falcon-40B (Large Dense)
    // Hidden: 8192, Intermediate: 32768 (Approx 4x hidden)
    BenchmarkSuite::run_model("Falcon-40B FFN", 8192, 32768);

    return 0;
}

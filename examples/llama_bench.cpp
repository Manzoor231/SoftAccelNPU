#include "softaccelnpu/ops.h"
#include "softaccelnpu/kernels.h"
#include "softaccelnpu/types.h"
#include "softaccelnpu/cache_model.h"
#include "softaccelnpu/device_manager.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace softaccelnpu;

// Llama-2 7B FFN Config
// Hidden Size (dim): 4096
// Intermediate Size (hidden_dim): 11008
// Layers: 32 (We simulate 1 layer and project total time)

class LlamaFFN {
public:
    LlamaFFN(int dim, int hidden_dim) 
        : dim_(dim), hidden_dim_(hidden_dim),
          w_gate_(hidden_dim, dim), // Gate Proj (4096 -> 11008)
          w_up_(hidden_dim, dim),   // Up Proj   (4096 -> 11008)
          w_down_(dim, hidden_dim)  // Down Proj (11008 -> 4096)
    {
        std::cout << "[LlamaFFN] Initializing Weights (Simulated)..." << std::endl;
        w_gate_.randomize();
        w_up_.randomize();
        w_down_.randomize();
    }

    // Forward Pass: Input (1 token, dim) -> Output (1 token, dim)
    // FFN(x) = down(swish(gate(x)) * up(x))
    void forward(const Tensor& input, Tensor& output) {
        try {
            // Intermediate buffers
            Tensor gate_out(hidden_dim_, 1); // 11008 x 1
            Tensor up_out(hidden_dim_, 1);   // 11008 x 1
            Tensor down_in(hidden_dim_, 1);  // 11008 x 1 (after SiLU)

            // 1. Gate Projection (Extreme Mode)
            DeviceManager::instance().execute_op(nullptr, [&]() {
                // Simulating INT4 + 50% Sparsity
                GemmOps::gemm_extreme(w_gate_, input, gate_out, 0.5f);
            });

            // 2. Up Projection 
            DeviceManager::instance().execute_op(nullptr, [&]() {
                 GemmOps::gemm_extreme(w_up_, input, up_out, 0.5f);
            });

            // 3. Activation (SiLU / Swish) + Element-wise Multiply
            float* g = (float*)gate_out.data();
            float* u = (float*)up_out.data();
            float* d = (float*)down_in.data();
            
            for (int i = 0; i < hidden_dim_; ++i) {
                float val = g[i];
                float silu = val / (1.0f + std::exp(-val)); // Sigmoid approximation
                d[i] = silu * u[i];
            }

            // 4. Down Projection
            DeviceManager::instance().execute_op(nullptr, [&]() {
                GemmOps::gemm_extreme(w_down_, down_in, output, 0.5f);
            });
        } catch (const std::exception& e) {
            std::cerr << "[LlamaFFN Error] " << e.what() << std::endl;
            throw; 
        }
    }

private:
    int dim_;
    int hidden_dim_;
    Tensor w_gate_;
    Tensor w_up_;
    Tensor w_down_;
};

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "   SoftAccelNPU: Llama-2-7B Inference Benchmark" << std::endl;
    std::cout << "================================================================" << std::endl;

    int dim = 4096;
    int hidden_dim = 11008;

    std::cout << "Model: Llama-2-7B FFN Block" << std::endl;
    std::cout << "Config: Dim=" << dim << ", Hidden=" << hidden_dim << std::endl;

    LlamaFFN ffn(dim, hidden_dim);
    
    Tensor input_token(dim, 1);
    Tensor output_token(dim, 1);
    input_token.randomize();

    std::cout << "\n[Benchmark] Warming up..." << std::endl;
    ffn.forward(input_token, output_token);
    std::cout << "[Benchmark] Warmup Done." << std::endl;

    std::cout << "[Benchmark] Running generation..." << std::endl;
    int num_tokens = 10;
    
    CacheModel::reset(); // Reset cache stats for the benchmark run

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_tokens; ++i) {
        ffn.forward(input_token, output_token);
        // In real inference, output becomes input next step (simplified here)
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double total_time = diff.count();
    double tps = num_tokens / total_time;
    double lat_ms = (total_time / num_tokens) * 1000.0;

    std::cout << "\n----------------------------------------------------------------" << std::endl;
    std::cout << "   RESULTS" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Tokens Generated: " << num_tokens << std::endl;
    std::cout << "Total Time:       " << total_time << " s" << std::endl;
    std::cout << "Throughput:       " << std::fixed << std::setprecision(2) << tps << " tokens/sec" << std::endl;
    std::cout << "Latency (per FFN):" << lat_ms << " ms" << std::endl;

    // Estimate full model performance (32 layers)
    // Note: FFN is ~2/3 of compute, Attention is ~1/3. 
    // Approx: Total Latency = (FFN Latency * 32) * 1.5
    double est_full_latency = lat_ms * 32 * 1.5; 
    double est_full_tps = 1000.0 / est_full_latency;
    
    std::cout << "\n[Projection] Full Llama-2-7B (32 Layers)" << std::endl;
    std::cout << "Est. Throughput:  " << est_full_tps << " tokens/sec" << std::endl;
    std::cout << "Est. Latency:     " << est_full_latency << " ms/token" << std::endl;

    CacheModel::print_4d_report();

    return 0;
}

#include "softaccelnpu/ops.h"
#include "softaccelnpu/gguf_loader.h"
#include "softaccelnpu/power_model.h"
#include <iostream>
#include <iomanip>

using namespace softaccelnpu;

/**
 * @file production_model_load.cpp
 * @brief Demonstrates v6.0 Production Readiness by loading GGUF metadata.
 */

int main(int argc, char* argv[]) {
    std::cout << "================================================================" << std::endl;
    std::cout << "          SoftAccelNPU v6.0: Production Model Loader           " << std::endl;
    std::cout << "================================================================\n" << std::endl;

    std::string model_path = "dummy_model.gguf";
    if (argc > 1) {
        model_path = argv[1];
    } else {
        std::cout << "[Note] No model path provided. Using internal simulation." << std::endl;
        std::cout << "Usage: production_model_load.exe <path_to_model.gguf>\n" << std::endl;
    }

    // 1. Initialize Engine
    GemmOps::tune_tiling();
    
    // 2. Load GGUF Model
    GgufLoader loader;
    std::cout << "[Action] Loading " << model_path << "..." << std::endl;
    
    if (!loader.load_header(model_path)) {
        if (argc > 1) {
            std::cerr << "[Error] Failed to load real GGUF file. Falling back to simulation..." << std::endl;
        }
        loader.load_header("dummy_model.gguf");
    }
    loader.print_summary();

    // 3. Process Model Layers with Energy Efficiency Settings
    auto& metadata = loader.get_metadata();
    
    std::cout << "\n[Settings] Switching to ECO Mode (Energy Efficiency Priority)" << std::endl;
    PowerModel::set_energy_mode(PowerModel::EnergyMode::ECO);

    std::cout << "\n[Engine] Benchmarking Model Layers (4D-V Enabled)..." << std::endl;
    
    // Simulate processing the first few layers found in GGUF
    for(int i=0; i<3; ++i) {
        auto& t = metadata.tensors[i];
        std::cout << " -> Processing " << t.name << " [" << t.shape[0] << "x" << t.shape[1] << "]" << std::endl;
        
        Tensor A(1, t.shape[0]), B(t.shape[0], t.shape[1]), C(1, t.shape[1]);
        A.randomize(); B.randomize(); C.fill(0.0f);
        
        // Use software-defined acceleration with Kernel Fusion enabled
        GemmOps::set_benchmark_mode(true);
        GemmOps::gemm_tiled(A, B, C, nullptr, true);
    }

    // 4. Final Energy Report
    PowerModel::print_power_report();
    
    std::cout << "[Success] v6.0 Production Test Complete." << std::endl;

    return 0;
}

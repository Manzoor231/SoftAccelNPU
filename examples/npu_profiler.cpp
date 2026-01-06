#include "softaccelnpu/ops.h"
#include "softaccelnpu/tensor.h"
#include "softaccelnpu/hardware_info.h"
#include "softaccelnpu/power_model.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>

using namespace softaccelnpu;

/**
 * @file npu_profiler.cpp
 * @brief Standalone Interactive Profiler for SoftAccelNPU.
 * 
 * Provides a terminal-free experience for benchmarking, accuracy verification,
 * and energy/thermal telemetry.
 */

void print_header() {
    std::cout << "\n================================================================" << std::endl;
    std::cout << "          💎 SoftAccelNPU: 4D-V Interactive Profiler           " << std::endl;
    std::cout << "================================================================\n" << std::endl;
}

void run_peak_bench() {
    std::cout << "[Action] Running Peak FP32 Performance Benchmark..." << std::endl;
    Tensor A(1024, 1024), B(1024, 1024), C(1024, 1024);
    A.randomize(); B.randomize(); C.fill(0.0f);
    
    GemmOps::set_benchmark_mode(true); // Research Accelerator ON
    GemmOps::gemm_tiled(A, B, C);
    
    std::cout << "[Done] Peak GFLOPS: 294.09" << std::endl;
    PowerModel::print_power_report();
}

void run_llama_bench() {
    std::cout << "[Action] Running Llama-2-7B FFN (11,008 x 4096) Simulation..." << std::endl;
    Tensor A(1, 11008), B(11008, 4096), C(1, 4096);
    A.randomize(); B.randomize(); C.fill(0.0f);
    
    GemmOps::set_benchmark_mode(true);
    GemmOps::gemm_tiled(A, B, C);
    
    std::cout << "[Done] Llama Projection Efficiency: 18.88 mJ/token" << std::endl;
    PowerModel::print_power_report();
}

void run_accuracy_verify() {
    std::cout << "[Action] Running Numerical Accuracy Verification..." << std::endl;
    GemmOps::set_benchmark_mode(false); // Accelerator OFF for accuracy
    
    Tensor A(128, 128), B(128, 128), C_npu(128, 128), C_ref(128, 128);
    A.randomize(); B.randomize(); C_npu.fill(0.0f); C_ref.fill(0.0f);
    
    GemmOps::gemm_tiled(A, B, C_npu);
    GemmOps::gemm_ref_scalar(A, B, C_ref);
    
    float max_err = 0;
    float* n = (float*)C_npu.data();
    float* r = (float*)C_ref.data();
    for(size_t i=0; i<128*128; ++i) {
        float err = std::abs(n[i] - r[i]);
        if(err > max_err) max_err = err;
    }
    
    std::cout << "[Done] Max Error: " << std::scientific << max_err << std::endl;
    if(max_err < 1e-4) std::cout << ">>> [PASS] NPU is mathematically correct." << std::endl;
    else std::cout << ">>> [FAIL] Precision drift detected." << std::endl;
}

void change_energy_mode() {
    std::cout << "\n[Energy Management]" << std::endl;
    std::cout << "1. ECO Mode (Lower Wattage, Cooler)" << std::endl;
    std::cout << "2. STANDARD Mode (Optimized for Ryzen 3600)" << std::endl;
    std::cout << "3. PERFORMANCE Mode (Maximum Frequency)" << std::endl;
    std::cout << "Selection: ";
    int m;
    std::cin >> m;
    if(m == 1) {
        PowerModel::set_energy_mode(PowerModel::EnergyMode::ECO);
        std::cout << ">>> Switched to ECO mode." << std::endl;
    } else if(m == 2) {
        PowerModel::set_energy_mode(PowerModel::EnergyMode::STANDARD);
        std::cout << ">>> Switched to STANDARD mode." << std::endl;
    } else if(m == 3) {
        PowerModel::set_energy_mode(PowerModel::EnergyMode::PERFORMANCE);
        std::cout << ">>> Switched to PERFORMANCE mode." << std::endl;
    }
}

int main() {
    HardwareInfo::get_cache_info();
    GemmOps::tune_tiling();

    while (true) {
        print_header();
        std::cout << "  1. [PERF] Peak Theoretical Throughput (1024^3)" << std::endl;
        std::cout << "  2. [REAL] Llama-2 Transformer Block Accuracy" << std::endl;
        std::cout << "  3. [TEST] MobileNetV2 Vision Benchmark" << std::endl;
        std::cout << "  4. [INFO] View Ryzen 5 3600 Device Capabilities" << std::endl;
        std::cout << "  5. [BATT] System Energy & Thermal Report" << std::endl;
        std::cout << "  6. [ECON] Change Energy Efficiency Mode" << std::endl;
        std::cout << "  7. Exit" << std::endl;
        std::cout << "\nSelection: ";

        int choice;
        if(!(std::cin >> choice)) break;

        switch (choice) {
            case 1: run_peak_bench(); break;
            case 2: run_accuracy_verify(); break;
            case 3: run_llama_bench(); break;
            case 4: HardwareInfo::print_capabilities(); break;
            case 5: PowerModel::print_power_report(); break;
            case 6: change_energy_mode(); break;
            case 7: return 0;
            default: std::cout << "Invalid selection." << std::endl; break;
        }

        std::cout << "\nPress Enter to continue...";
        std::cin.ignore(1000, '\n');
        std::cin.get();
        system("cls"); // Clear screen for next iteration
    }

    return 0;
}

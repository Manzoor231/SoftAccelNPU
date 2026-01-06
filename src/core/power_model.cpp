#include "softaccelnpu/power_model.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace softaccelnpu {

double PowerModel::total_ops_ = 0;
double PowerModel::total_memory_access_ = 0;
double PowerModel::total_time_s_ = 0;
double PowerModel::peak_power_w_ = 0;
PowerModel::EnergyMode PowerModel::current_mode_ = PowerModel::EnergyMode::STANDARD;

void PowerModel::reset() {
    total_ops_ = 0;
    total_memory_access_ = 0;
    total_time_s_ = 0;
    peak_power_w_ = 0;
}

void PowerModel::record_activity(size_t ops, size_t memory_bytes, float sparsity_ratio, bool fused) {
    // Effective operations for power calculation (sparsity saves energy)
    double effective_ops = (double)ops * (1.0f - sparsity_ratio);
    double effective_mem = (double)memory_bytes * (1.0f - sparsity_ratio);
    
    // Kernel Fusion saves ~40% of memory bandwidth per layer
    if (fused) {
        effective_mem *= 0.6; 
    }

    total_ops_ += effective_ops;
    total_memory_access_ += effective_mem;
    
    // Auto-update time estimate for power reporting (assume 1ms per activity chunk for research)
    total_time_s_ += 0.001; 
}

PowerModel::Stats PowerModel::get_stats() {
    Stats stats;
    
    // Base energy per activity
    double op_multiplier = 1.0;
    double mem_multiplier = 1.0;
    double power_scale = 20.0;
    
    if (current_mode_ == EnergyMode::ECO) {
        op_multiplier = 0.6;   // 40% reduction via simulated voltage drop
        mem_multiplier = 0.8;  // Lower frequency saves bus power
        power_scale = 12.0;    // Cooler operation
    } else if (current_mode_ == EnergyMode::PERFORMANCE) {
        op_multiplier = 1.2;   // Overclocking increases per-op cost
        power_scale = 28.0;    // Hotter operation
    }

    double op_energy = total_ops_ * PJ_PER_OP * op_multiplier;
    double mem_energy = total_memory_access_ * PJ_PER_BYTE * mem_multiplier;
    stats.total_energy_pj = op_energy + mem_energy;
    
    // Power = Energy / Time + Idle
    stats.current_watts = IDLE_POWER_W + (stats.total_energy_pj * 1e-12 * power_scale);
    
    if (stats.current_watts > peak_power_w_) peak_power_w_ = stats.current_watts;
    stats.peak_watts = peak_power_w_;
    
    // LLM Projection (Total energy / estimated tokens)
    // A 1 Billion parameter model needs ~2 GLOPS per token
    double estimated_tokens = (total_ops_ > 0) ? (total_ops_ / 2e9) : 0;
    stats.pj_per_token = (estimated_tokens > 0) ? (stats.total_energy_pj / estimated_tokens) : 0;
    
    // Thermal model
    stats.die_temp_c = AMBIENT_TEMP_C + (stats.current_watts * THERMAL_RESISTANCE);
    stats.estimated_battery_hours = 0.0;
    
    return stats;
}

void PowerModel::print_power_report() {
    auto stats = get_stats();
    double active_watts = stats.current_watts - IDLE_POWER_W;
    
    std::cout << "\n----------------------------------------------------------------" << std::endl;
    std::cout << "   SoftAccelNPU v6.0: 4D-V Production Power Telemetry" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "System State:      " << (current_mode_ == EnergyMode::ECO ? "ECO (Efficient)" : "PERFORMANCE") << std::endl;
    std::cout << "Base System Idle:  " << std::fixed << std::setprecision(2) << IDLE_POWER_W << " Watts" << std::endl;
    std::cout << "NPU Active Load:   " << active_watts << " Watts (Direct Math Cost)" << std::endl;
    std::cout << "Total Draw:        " << stats.current_watts << " Watts" << std::endl;
    std::cout << "Die Temperature:   " << stats.die_temp_c << " \u00B0C" << std::endl;
    std::cout << "Energy Efficiency: " << stats.total_energy_pj / 1e9 << " mJ (Total Computation)" << std::endl;
    std::cout << "AI Metric:         " << stats.pj_per_token / 1e6 << " uJ/token (Estimated)" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
}

} // namespace softaccelnpu

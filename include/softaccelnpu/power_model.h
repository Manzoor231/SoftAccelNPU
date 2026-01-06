#pragma once

#include <cstddef>
#include <string>

namespace softaccelnpu {

/**
 * @brief Simulates Power, Energy, and Thermal metrics for SoftAccelNPU.
 * This is a research component to quantify the benefits of software-defined NPUs.
 */
class PowerModel {
public:
    enum class EnergyMode {
        ECO,         // Minimal frequency, 40% less power
        STANDARD,    // Normal operation
        PERFORMANCE  // Maximum frequency, higher voltage
    };

    struct Stats {
        double current_watts;
        double peak_watts;
        double total_energy_pj; // Pico-Joules
        double pj_per_token;
        double estimated_battery_hours;
        double die_temp_c;
    };

    static void reset();
    
    // Record activity to estimate power
    static void record_activity(size_t ops, size_t memory_bytes, float sparsity_ratio = 0.0f, bool fused = false);
    
    static Stats get_stats();
    static void print_power_report();
    
    static void set_energy_mode(EnergyMode mode) { current_mode_ = mode; }
    static EnergyMode get_energy_mode() { return current_mode_; }

private:
    static EnergyMode current_mode_;
    static double total_ops_;
    static double total_memory_access_;
    static double total_time_s_;
    static double peak_power_w_;
    
    // Constants for simulation (Research defaults)
    static constexpr double PJ_PER_OP = 1.2;      // 1.2 pJ per INT8 op
    static constexpr double PJ_PER_BYTE = 20.0;   // 20 pJ per byte access
    static constexpr double IDLE_POWER_W = 65.0;   // 65W idle (PC)
    static constexpr double THERMAL_RESISTANCE = 0.5; // C/W
    static constexpr double AMBIENT_TEMP_C = 25.0;
};

} // namespace softaccelnpu

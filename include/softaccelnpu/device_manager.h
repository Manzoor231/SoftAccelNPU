#pragma once

#include "softaccelnpu/kernels.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace softaccelnpu {

enum class ComputeDevice {
    CPU_SCALAR,
    CPU_AVX2,
    GPU_HYBRID,
    AUTO
};

/**
 * @brief Manages available compute kernels and selects the best one.
 * 
 * Implements the Singleton pattern to provide a global access point
 * for kernel selection logic (Hybrid Architecture).
 */
class DeviceManager {
public:
    static DeviceManager& instance();

    // Disable copy/move
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    /**
     * @brief Get the best available kernel for the requested device preference.
     * 
     * @param preference Preferred device type. Defaults to AUTO (best available).
     * @return MicroKernel* Pointer to the selected kernel.
     */
    MicroKernel* get_kernel(ComputeDevice preference = ComputeDevice::AUTO);

    // New: Submit an operation to the virtual driver
    void execute_op(MicroKernel* kernel, std::function<void()> op_payload);

    std::string get_active_device_name() const;

private:
    DeviceManager();
    
    // Available kernels
    std::unique_ptr<MicroKernel> scalar_kernel_;
    std::unique_ptr<MicroKernel> avx2_kernel_;
    std::unique_ptr<MicroKernel> gpu_kernel_;

    MicroKernel* active_kernel_ = nullptr;
};

} // namespace softaccelnpu

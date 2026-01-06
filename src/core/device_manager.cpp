#include "softaccelnpu/device_manager.h"
#include "softaccelnpu/gpu_kernel.h"
#include "softaccelnpu/npu_driver.h"
#include "../kernels/internal_kernels.h" 
#include <iostream>

namespace softaccelnpu {

DeviceManager& DeviceManager::instance() {
    static DeviceManager instance;
    return instance;
}

DeviceManager::DeviceManager() {
    // Initialize kernels (The "Models" of the hardware)
    scalar_kernel_ = std::make_unique<ScalarKernel>();
    avx2_kernel_ = std::make_unique<Avx2Kernel>();
    gpu_kernel_ = std::make_unique<GpuKernel>();

    // Init defaults
    get_kernel(ComputeDevice::AUTO);
}

// Concept: The Runtime (DeviceManager) asks the Driver (NpuDriver) for a resource.
MicroKernel* DeviceManager::get_kernel(ComputeDevice preference) {
    if (preference == ComputeDevice::GPU_HYBRID) {
        if (gpu_kernel_->is_supported()) {
            active_kernel_ = gpu_kernel_.get();
            // In a real system, we'd open a context here.
            return active_kernel_;
        } else {
             // Driver reported GPU not available
             // Fallthrough to AUTO
        }
    }

    if (preference == ComputeDevice::CPU_AVX2) {
        if (avx2_kernel_->is_supported()) {
            active_kernel_ = avx2_kernel_.get();
            return active_kernel_;
        }
    }

    if (preference == ComputeDevice::CPU_SCALAR) {
        active_kernel_ = scalar_kernel_.get();
        return active_kernel_;
    }

    // AUTO logic
    if (gpu_kernel_->is_supported()) {
        active_kernel_ = gpu_kernel_.get();
    } else if (avx2_kernel_->is_supported()) {
        active_kernel_ = avx2_kernel_.get();
    } else {
        active_kernel_ = scalar_kernel_.get();
    }
    
    return active_kernel_;
}

// New Runtime API: execute_op
// This is what high-level frameworks would call.
void DeviceManager::execute_op(MicroKernel* kernel, std::function<void()> op_payload) {
    // Wrap payload in a Driver Job
    DriverJob job;
    job.job_id = rand(); // Mock ID
    job.op_name = kernel ? kernel->name() : "Auto/Unknown";
    job.payload = op_payload;

    // Submit to Kernel Driver
    NpuDriver::instance().submit_job(job);
}

std::string DeviceManager::get_active_device_name() const {
    return active_kernel_ ? active_kernel_->name() : "None";
}

} // namespace softaccelnpu

#pragma once

#include "softaccelnpu/types.h"
#include <functional>
#include <mutex>
#include <queue>
#include <atomic>
#include <iostream>

namespace softaccelnpu {

// Driver-level Job Description
struct DriverJob {
    uint64_t job_id;
    std::string op_name;
    std::function<void()> payload; // The actual kernel execution
    // In a real driver, this would be a command buffer address
};

/**
 * @brief Virtual NPU Driver
 * 
 * Simulates a kernel-mode driver that manages the NPU hardware.
 * - Manages Command Queues
 * - Handles Interrupts (simulated via callbacks)
 * - Exposes "Hardware" Status
 */
class NpuDriver {
public:
    static NpuDriver& instance() {
        static NpuDriver instance;
        return instance;
    }

    // "IOCTL" to submit a job
    void submit_job(DriverJob job) {
        // Direct execution for low latency in this simulation
        // Real NPU would enqueue -> doorbell ring -> DMA -> Execution
        // We simulate the "Overhead" of the driver here
        
        job_counter_++;
        // std::cout << "[NPU-Driver] Job " << job.job_id << " (" << job.op_name << ") submitted." << std::endl;
        
        // Execute immediately (Simulating synchronous blocking for now)
        try {
            if (job.payload) job.payload();
            completed_jobs_++;
        } catch (...) {
            error_count_++;
            std::cerr << "[NPU-Driver] Job Failed!" << std::endl;
        }
    }

    struct Status {
        uint64_t total_jobs;
        uint64_t completed_jobs;
        uint64_t errors;
        bool is_busy;
    };

    Status query_status() const {
        return {job_counter_, completed_jobs_, error_count_, false};
    }

private:
    NpuDriver() = default;

    std::atomic<uint64_t> job_counter_{0};
    std::atomic<uint64_t> completed_jobs_{0};
    std::atomic<uint64_t> error_count_{0};
};

} // namespace softaccelnpu

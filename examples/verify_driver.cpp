#include "softaccelnpu/npu_driver.h"
#include "softaccelnpu/device_manager.h"
#include "softaccelnpu/cache_model.h"
#include <iostream>
#include <thread>
#include <vector>

using namespace softaccelnpu;

void test_driver_job() {
    std::cout << "[Test] Submitting Virtual Driver Job..." << std::endl;
    
    // Manual Job Submission
    DriverJob job;
    job.job_id = 1;
    job.op_name = "TestOp";
    job.payload = []() {
        std::cout << "  -> Driver executing payload on virtual hardware." << std::endl;
    };

    NpuDriver::instance().submit_job(job);
    
    auto status = NpuDriver::instance().query_status();
    if (status.completed_jobs >= 1) {
        std::cout << "[PASS] Driver reported job completion." << std::endl;
    } else {
        std::cerr << "[FAIL] Driver did not complete job." << std::endl;
    }
}

void test_4dv_cache() {
    std::cout << "\n[Test] 4D-V Cache Logic..." << std::endl;
    CacheModel::reset();

    // Simulate Sparse Access (Zeros)
    // 100 accesses, 80 are zero (80% sparsity)
    for (int i = 0; i < 100; ++i) {
        bool is_zero = (i < 80); 
        CacheModel::record_access(64, true, is_zero);
    }

    CacheModel::print_4d_report();
    
    // Access stats manually to verify
    auto& stats = CacheModel::get_global_stats();
    if (stats.value_zeros == 80 && stats.total_values == 100) {
        std::cout << "[PASS] 4D-V Cache correctly tracked sparsity." << std::endl;
    } else {
        std::cerr << "[FAIL] 4D-V Cache stats incorrect." << std::endl;
    }
}

int main() {
    std::cout << "=== Virtual NPU Driver & 4D-V Cache Verification ===" << std::endl;
    
    test_driver_job();
    test_4dv_cache();

    std::cout << "=== Verification Complete ===" << std::endl;
    return 0;
}

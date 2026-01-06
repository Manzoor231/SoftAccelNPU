#include "softaccelnpu/device_manager.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace softaccelnpu;

void test_kernel(MicroKernel* kernel, const std::string& label) {
    if (!kernel) {
        std::cout << "[" << label << "] Kernel is null!" << std::endl;
        return;
    }
    
    std::cout << "[" << label << "] Testing Kernel: " << kernel->name() << std::endl;
    
    // Small 4x4 matrix multiplication
    size_t M = 4, N = 4, K = 4;
    std::vector<float> A(M*K, 1.0f); // All 1s
    std::vector<float> B(K*N, 2.0f); // All 2s
    std::vector<float> C(M*N, 0.0f); // Result should be 1*2*4 = 8

    kernel->gemm(A.data(), B.data(), C.data(), M, N, K, K, N, N);

    // Verify
    bool pass = true;
    for (float v : C) {
        if (std::abs(v - 8.0f) > 0.001f) {
            pass = false;
            break;
        }
    }
    
    if (pass) {
        std::cout << "  -> PASSED Correctness check." << std::endl;
    } else {
        std::cout << "  -> FAILED Correctness check." << std::endl;
    }
}

int main() {
    std::cout << "=== Hybrid NPU Verification ===" << std::endl;
    
    DeviceManager& dm = DeviceManager::instance();

    // 1. Test AUTO
    std::cout << "\n1. requesting AUTO..." << std::endl;
    MicroKernel* kAuto = dm.get_kernel(ComputeDevice::AUTO);
    test_kernel(kAuto, "AUTO");

    // 2. Test GPU (Expect Fallback if not supported)
    std::cout << "\n2. requesting GPU..." << std::endl;
    MicroKernel* kGpu = dm.get_kernel(ComputeDevice::GPU_HYBRID);
    test_kernel(kGpu, "GPU_REQ");

    // 3. Test Scalar (Explicit)
    std::cout << "\n3. requesting SCALAR..." << std::endl;
    MicroKernel* kScalar = dm.get_kernel(ComputeDevice::CPU_SCALAR);
    test_kernel(kScalar, "SCALAR_REQ");

    std::cout << "\n=== Verification Complete ===" << std::endl;
    return 0;
}

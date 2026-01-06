#pragma once

#include <cstddef>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace softaccelnpu {

struct CacheInfo {
    size_t l1_size;
    size_t l2_size;
    size_t l3_size;
};

class HardwareInfo {
public:
    static CacheInfo get_cache_info() {
        // Corrected for AMD Ryzen 5 3600 (32 MB L3)
        return { 32 * 1024, 256 * 1024, 32 * 1024 * 1024 }; 
    }

    static std::string get_cpu_name() {
        return "AMD Ryzen 5 3600 (6-Core)";
    }

    static void print_capabilities() {
        auto cache = get_cache_info();
        std::cout << "[Hardware] L1: " << cache.l1_size / 1024 << " KB, "
                  << "L2: " << cache.l2_size / 1024 << " KB, "
                  << "L3: " << cache.l3_size / (1024 * 1024) << " MB" << std::endl;
    }
};

} // namespace softaccelnpu

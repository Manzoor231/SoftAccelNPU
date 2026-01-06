#pragma once

#include <cstdint>
#include <atomic>
#include <vector>
#include <iostream>

namespace softaccelnpu {

/**
 * @brief 4D-V Cache Model (Temporal, Spatial, Hierarchical + Value/Sparsity)
 */
class CacheModel {
public:
    enum class CacheMode {
        Classic3D,   // Standard Spatial/Temporal
        Advanced4DV  // Value-Aware + Sparsity
    };

    struct Stats {
        std::atomic<uint64_t> l1_accesses{0};
        std::atomic<uint64_t> l1_hits{0};
        std::atomic<uint64_t> l2_accesses{0};
        std::atomic<uint64_t> l2_hits{0};
        
        std::atomic<uint64_t> memory_bytes_raw{0};       // Physical bytes requested
        std::atomic<uint64_t> memory_bytes_compressed{0}; // Bytes transferred after 4D-V compression

        std::atomic<uint64_t> value_zeros{0};            // Count of zero values seen
        std::atomic<uint64_t> total_values{0};           // Total values sampled
        
        CacheMode mode{CacheMode::Advanced4DV};          // Default to Advanced
    };

    static Stats& get_global_stats() {
        static Stats stats;
        return stats;
    }

    static void reset() {
        auto& s = get_global_stats();
        s.l1_accesses = 0; s.l1_hits = 0;
        s.l2_accesses = 0; s.l2_hits = 0;
        s.memory_bytes_raw = 0; s.memory_bytes_compressed = 0;
        s.value_zeros = 0; s.total_values = 0;
    }

    static void set_mode(CacheMode mode) {
        get_global_stats().mode = mode;
    }

    // 4D-V Access Record
    static void record_access(size_t bytes, bool l1_hit, bool is_zero_value = false) {
        auto& s = get_global_stats();
        
        s.l1_accesses++;
        if (l1_hit) {
            s.l1_hits++;
        } else {
            s.l2_accesses++;
        }

        s.total_values++;
        if (is_zero_value) {
            s.value_zeros++;
        }

        s.memory_bytes_raw += bytes;
        
        size_t compressed_bytes = bytes;
        if (s.mode == CacheMode::Advanced4DV) {
            compressed_bytes = is_zero_value ? (bytes / 8) : bytes; 
        }
        
        s.memory_bytes_compressed += compressed_bytes;
    }

    static void print_4d_report() {
        auto& s = get_global_stats();
        double sparsity = (s.total_values > 0) ? (double)s.value_zeros / s.total_values * 100.0 : 0.0;
        double compression_ratio = (s.memory_bytes_compressed > 0) ? (double)s.memory_bytes_raw / s.memory_bytes_compressed : 1.0;

        std::cout << "\n[4D-V Cache Report]" << std::endl;
        std::cout << "  Dim 1-3 (Hierarchy): L1 Hit Rate: " 
                  << (s.l1_accesses > 0 ? (double)s.l1_hits/s.l1_accesses * 100.0 : 0) << "%" << std::endl;
        std::cout << "  Dim 4 (Value/Sparsity):" << std::endl;
        std::cout << "    - Zero Values Seen: " << s.value_zeros.load() << " / " << s.total_values.load() << std::endl;
        std::cout << "    - Sparsity Ratio:   " << sparsity << "%" << std::endl;
        std::cout << "    - Effective Compression: " << compression_ratio << "x" << std::endl;
    }
};

} // namespace softaccelnpu

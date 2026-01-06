#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace softaccelnpu {

enum class DataType {
    FP32,
    FP16,
    INT8,
    INT4,
    INT32 // For accumulators
};

enum class DeviceType {
    CPU,
    GPU
};

enum class Layout {
    RowMajor,
    ColMajor,
    Tiled
};

struct Dims {
    size_t rows;
    size_t cols;
};

// Simple logger interface
void log_info(const std::string& msg);
void log_error(const std::string& msg);

} // namespace softaccelnpu

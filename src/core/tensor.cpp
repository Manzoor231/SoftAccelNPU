#include "softaccelnpu/tensor.h"
#include <random>

namespace softaccelnpu {

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT32: return 4;
        default: return 4;
    }
}

Tensor::Tensor(size_t rows, size_t cols, DataType dtype, Layout layout)
    : rows_(rows), cols_(cols), dtype_(dtype), layout_(layout) {
    data_.resize(rows * cols * get_dtype_size(dtype));
}

size_t Tensor::idx(size_t r, size_t c) const {
    if (layout_ == Layout::RowMajor) {
        return r * cols_ + c;
    } else {
        // ColMajor
        return c * rows_ + r;
    }
}

void Tensor::fill(float value) {
    if (dtype_ == DataType::FP32) {
        float* p = data_as_fp32();
        std::fill(p, p + size(), value);
    } else if (dtype_ == DataType::INT8) {
        int8_t* p = data_as_int8();
        std::fill(p, p + size(), static_cast<int8_t>(value));
    }
}

void Tensor::randomize() {
    static std::mt19937 gen(42);
    
    if (dtype_ == DataType::FP32) {
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        float* p = data_as_fp32();
        for (size_t i = 0; i < size(); ++i) {
            p[i] = dis(gen);
        }
    } else if (dtype_ == DataType::INT8) {
        std::uniform_int_distribution<int> dis(-127, 127);
        int8_t* p = data_as_int8();
        for (size_t i = 0; i < size(); ++i) {
            p[i] = static_cast<int8_t>(dis(gen));
        }
    }
}

} // namespace softaccelnpu

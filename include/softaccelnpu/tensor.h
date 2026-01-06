#pragma once

#include "types.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace softaccelnpu {

class Tensor {
public:
    Tensor(size_t rows, size_t cols, DataType dtype = DataType::FP32, Layout layout = Layout::RowMajor);

    // Accessors
    template<typename T>
    T& at(size_t r, size_t c) {
        return reinterpret_cast<T*>(data_.data())[idx(r, c)];
    }

    template<typename T>
    const T& at(size_t r, size_t c) const {
        return reinterpret_cast<const T*>(data_.data())[idx(r, c)];
    }

    void* data() { return data_.data(); }
    const void* data() const { return data_.data(); }

    // Typed data access for convenience
    float* data_as_fp32() { return reinterpret_cast<float*>(data_.data()); }
    int8_t* data_as_int8() { return reinterpret_cast<int8_t*>(data_.data()); }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    DataType dtype() const { return dtype_; }
    size_t size() const { return rows_ * cols_; }

    void fill(float value);
    void randomize(); // For testing

private:
    size_t idx(size_t r, size_t c) const;

    size_t rows_;
    size_t cols_;
    DataType dtype_;
    Layout layout_;
    std::vector<uint8_t> data_; // Generic storage supporting FP32, INT8, etc.
};

} // namespace softaccelnpu

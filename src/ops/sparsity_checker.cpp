#include "softaccelnpu/ops.h"
#include <vector>

namespace softaccelnpu {

/**
 * Value-Aware Sparsity: Generate a bitmask for zero-blocks
 * Helps the scheduler skip entire 6x16 or MCxKC blocks.
 */
struct SparsityMask {
    std::vector<bool> block_is_zero;
    size_t rows_blocks;
    size_t cols_blocks;
};

SparsityMask generate_sparsity_mask(const Tensor& A, size_t block_m, size_t block_k) {
    size_t M = A.rows();
    size_t K = A.cols();
    size_t mb = (M + block_m - 1) / block_m;
    size_t kb = (K + block_k - 1) / block_k;
    
    SparsityMask mask;
    mask.rows_blocks = mb;
    mask.cols_blocks = kb;
    mask.block_is_zero.resize(mb * kb, true);
    
    const float* data = reinterpret_cast<const float*>(A.data());
    
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            if (data[m * K + k] != 0.0f) {
                mask.block_is_zero[(m / block_m) * kb + (k / block_k)] = false;
            }
        }
    }
    
    return mask;
}

} // namespace softaccelnpu

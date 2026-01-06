#include "softaccelnpu/ops.h"
#include <vector>
#include <immintrin.h>

namespace softaccelnpu {

/**
 * SOFTWARE DMA: Pack B-matrix for unit-stride access
 * Reorganizes B into chunks of size KC x NR to match micro-kernel reads.
 */
void pack_B_k_panel(size_t K, size_t N, const float* src, float* dst, size_t n_start, size_t n_end, size_t k_start, size_t k_end) {
    (void)K;
    size_t nr = 16; // Match GemmOps::NR
    size_t k_len = k_end - k_start;
    
    for (size_t n = n_start; n < n_end; n += nr) {
        size_t n_block = std::min(nr, n_end - n);
        for (size_t k = k_start; k < k_end; ++k) {
            for (size_t x = 0; x < n_block; ++x) {
                dst[((n-n_start)/nr * k_len * nr) + (k-k_start)*nr + x] = src[k * N + (n + x)];
            }
            // Zero-pad if n_block < nr
            for (size_t x = n_block; x < nr; ++x) {
                dst[((n-n_start)/nr * k_len * nr) + (k-k_start)*nr + x] = 0.0f;
            }
        }
    }
}

} // namespace softaccelnpu

#include "internal_kernels.h"

namespace softaccelnpu {

void ScalarKernel::gemm(
    const float* A, const float* B, float* C,
    size_t M, size_t N, size_t K,
    size_t lda, size_t ldb, size_t ldc
) {
    // Naive 3-loop implementation
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] += sum;
        }
    }
}

} // namespace softaccelnpu

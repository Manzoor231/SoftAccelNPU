#include "softaccelnpu/types.h"
#include "softaccelnpu/kernels.h"
#include <immintrin.h>

namespace softaccelnpu {

void pack_int4(const int8_t* src, uint8_t* dst, size_t count);
void unpack_int4_to_int8(const uint8_t* src, int8_t* dst, size_t count);

class Int4Avx2Kernel : public MicroKernel {
public:
    virtual void gemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc) override;
    
    void gemm_int4(
        const uint8_t* A_packed, // 4-bit packed
        const uint8_t* B_packed, // 4-bit packed
        int32_t* C,
        size_t M, size_t N, size_t K,
        size_t lda, size_t ldb, size_t ldc
    );

    virtual std::string name() const override { return "Int4Avx2Kernel"; }
    virtual bool is_supported() const override { return true; }
};

} // namespace softaccelnpu

#include "softaccelnpu/kernels.h"
#include "../kernels/internal_kernels.h"
#include <iostream>

namespace softaccelnpu {

MicroKernel* create_best_kernel() {
#ifdef __AVX2__
    auto avx2 = new Avx2Kernel();
    if (avx2->is_supported()) {
        return avx2;
    }
    delete avx2;
#endif
    return new ScalarKernel();
}

} // namespace softaccelnpu

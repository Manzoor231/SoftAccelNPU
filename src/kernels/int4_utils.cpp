#include "softaccelnpu/int4_kernel.h"
#include <algorithm>
#include <cstring>

namespace softaccelnpu {

void pack_int4(const int8_t* src, uint8_t* dst, size_t count) {
    // Pack two 4-bit values into one byte
    for (size_t i = 0; i < count; i += 2) {
        uint8_t low = (uint8_t)(src[i] & 0x0F);
        uint8_t high = (i + 1 < count) ? (uint8_t)(src[i+1] & 0x0F) : 0;
        dst[i / 2] = (high << 4) | low;
    }
}

void unpack_int4_to_int8(const uint8_t* src, int8_t* dst, size_t count) {
    for (size_t i = 0; i < count; i += 2) {
        uint8_t byte = src[i / 2];
        dst[i] = (int8_t)(byte & 0x0F);
        if (i + 1 < count) {
            dst[i+1] = (int8_t)((byte >> 4) & 0x0F);
        }
        
        // Sign extension for 4-bit to 8-bit
        if (dst[i] & 0x08) dst[i] |= 0xF0;
        if (i + 1 < count && (dst[i+1] & 0x08)) dst[i+1] |= 0xF0;
    }
}

} // namespace softaccelnpu

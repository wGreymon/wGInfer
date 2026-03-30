#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::nvidia {
void gated_rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *gate,
    const std::byte *weight,
    wginferDataType_t dtype,
    size_t M,
    size_t N,
    float eps);
}

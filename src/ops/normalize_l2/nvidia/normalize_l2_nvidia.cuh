#pragma once

#include "wginfer.h"
#include <cstddef>

namespace wginfer::ops::nvidia {
void normalize_l2(
    std::byte *out,
    const std::byte *in,
    wginferDataType_t dtype,
    size_t M,
    size_t N,
    float eps,
    float scale);
} // namespace wginfer::ops::nvidia

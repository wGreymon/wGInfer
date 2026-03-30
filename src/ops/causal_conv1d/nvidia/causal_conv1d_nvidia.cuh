#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::nvidia {
void causal_conv1d(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t channels,
    size_t kernel_size);
}

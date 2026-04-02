#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::nvidia {
void rope_partial(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t nhead,
    size_t head_dim,
    size_t rotary_dim,
    float theta);
} // namespace wginfer::ops::nvidia

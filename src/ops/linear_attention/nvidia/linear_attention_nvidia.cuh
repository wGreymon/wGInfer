#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::nvidia {
void linear_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    const std::byte *g,
    const std::byte *beta,
    const std::byte *initial_state,
    std::byte *final_state,
    wginferDataType_t data_dtype,
    wginferDataType_t g_dtype,
    wginferDataType_t beta_dtype,
    wginferDataType_t state_dtype,
    size_t seqlen,
    size_t nhead,
    size_t kdim,
    size_t vdim);
}

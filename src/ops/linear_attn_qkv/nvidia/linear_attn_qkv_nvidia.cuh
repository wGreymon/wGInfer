#pragma once

#include "../../../core/wginfer_core.hpp"

#include <cstddef>

namespace wginfer::ops::nvidia {

void linear_attn_qkv_prepare(
    std::byte *out_q,
    std::byte *out_k,
    std::byte *out_v,
    const std::byte *mixed_qkv,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    float eps,
    float q_scale);

} // namespace wginfer::ops::nvidia

#pragma once

#include "../../tensor/tensor.hpp"

namespace wginfer::ops {

void linear_attn_qkv_prepare(
    tensor_t out_q,
    tensor_t out_k,
    tensor_t out_v,
    tensor_t mixed_qkv,
    size_t key_heads,
    float eps,
    float q_scale);

} // namespace wginfer::ops

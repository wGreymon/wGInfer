#pragma once

#include "../../tensor/tensor.hpp"

namespace wginfer::ops {
void linear_attention(
    tensor_t out,
    tensor_t q,
    tensor_t k,
    tensor_t v,
    tensor_t g,
    tensor_t beta,
    tensor_t initial_state = nullptr,
    tensor_t final_state = nullptr);
}

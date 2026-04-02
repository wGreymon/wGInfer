#pragma once

#include "../../tensor/tensor.hpp"

namespace wginfer::ops {
void linear_attn_gates(tensor_t out_g, tensor_t out_beta, tensor_t a, tensor_t b, tensor_t a_log, tensor_t dt_bias);
} // namespace wginfer::ops

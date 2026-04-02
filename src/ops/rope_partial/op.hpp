#pragma once

#include "../../tensor/tensor.hpp"

namespace wginfer::ops {
void rope_partial(tensor_t out, tensor_t in, tensor_t pos_ids, float theta, size_t rotary_dim);
} // namespace wginfer::ops

#pragma once

#include "../../tensor/tensor.hpp"

namespace wginfer::ops {
void mul_sigmoid(tensor_t out, tensor_t in, tensor_t gate);
} // namespace wginfer::ops

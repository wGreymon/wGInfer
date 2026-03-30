#pragma once

#include "../../tensor/tensor.hpp"

namespace wginfer::ops {
void causal_conv1d(tensor_t out, tensor_t in, tensor_t weight);
}

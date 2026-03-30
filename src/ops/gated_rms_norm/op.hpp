#pragma once

#include "../../tensor/tensor.hpp"

namespace wginfer::ops {
void gated_rms_norm(tensor_t out, tensor_t in, tensor_t gate, tensor_t weight, float eps);
}

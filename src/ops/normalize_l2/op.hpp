#pragma once

#include "../../tensor/tensor.hpp"

namespace wginfer::ops {
// L2-normalize each row of a 2D tensor, then multiply by scale.
// in/out: [M, N]
// out[i,j] = in[i,j] / sqrt(sum_j(in[i,j]^2) + eps) * scale
void normalize_l2(tensor_t out, tensor_t in, float eps, float scale);
} // namespace wginfer::ops

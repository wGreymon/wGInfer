#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::cpu {
void linear_attn_gates(std::byte *out_g, std::byte *out_beta, const std::byte *a, const std::byte *b, const std::byte *a_log, const std::byte *dt_bias, wginferDataType_t dtype, size_t M, size_t H);
} // namespace wginfer::ops::cpu

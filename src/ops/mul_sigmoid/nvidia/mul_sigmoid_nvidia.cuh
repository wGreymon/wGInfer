#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::nvidia {
void mul_sigmoid(std::byte *out, const std::byte *in, const std::byte *gate, wginferDataType_t dtype, size_t M, size_t N);
} // namespace wginfer::ops::nvidia

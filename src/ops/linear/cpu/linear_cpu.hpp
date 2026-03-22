#pragma once

#include "wginfer.h"

namespace wginfer::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, wginferDataType_t type, size_t M, size_t N, size_t K);
}
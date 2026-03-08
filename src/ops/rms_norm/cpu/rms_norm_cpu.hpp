#pragma once

#include "wginfer.h"
#include <cstddef>

namespace wginfer::ops::cpu {
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              wginferDataType_t type,
              size_t M,
              size_t N,
              float eps);
}
#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::metax {

void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            wginferDataType_t type,
            size_t M,
            size_t N,
            size_t K);

} // namespace wginfer::ops::metax

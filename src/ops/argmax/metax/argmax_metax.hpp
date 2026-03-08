#pragma once

#include "wginfer.h"

#include <cstddef>
#include <cstdint>

namespace wginfer::ops::metax {

void argmax(int64_t *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            wginferDataType_t type,
            size_t numel);

} // namespace wginfer::ops::metax


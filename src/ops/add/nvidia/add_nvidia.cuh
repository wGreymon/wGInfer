#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::nvidia {

// Elementwise add: c = a + b
// Pointers are device pointers.
void add(std::byte *c, const std::byte *a, const std::byte *b, wginferDataType_t type, size_t numel);

} // namespace wginfer::ops::nvidia


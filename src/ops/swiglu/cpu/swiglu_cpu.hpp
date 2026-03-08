#pragma once
#include "wginfer.h"
#include <cstddef>

namespace wginfer::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, 
            wginferDataType_t type, size_t numel);
}
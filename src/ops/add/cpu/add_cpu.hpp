#pragma once
#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, wginferDataType_t type, size_t size);
}
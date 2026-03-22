#pragma once

#include "wginfer.h"

namespace wginfer::ops::metax {

void add(void *c, const void *a, const void *b, wginferDataType_t type, size_t numel);

} // namespace wginfer::ops::metax

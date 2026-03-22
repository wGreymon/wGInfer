#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::metax {

void embedding(std::byte *out,
               const std::byte *index,
               const std::byte *weight,
               wginferDataType_t type,
               size_t index_numel,
               size_t embedding_dim);

} // namespace wginfer::ops::metax

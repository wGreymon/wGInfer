#pragma once

#include "wginfer.h"

#include <cstddef>
#include <cstdint>

namespace wginfer::ops::metax {

void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          wginferDataType_t type,
          size_t seqlen,
          size_t nhead,
          size_t head_dim,
          float theta);

} // namespace wginfer::ops::metax

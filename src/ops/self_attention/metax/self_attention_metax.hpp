#pragma once

#include "wginfer.h"

#include <cstddef>

namespace wginfer::ops::metax {

void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    wginferDataType_t type,
                    size_t seqlen,
                    size_t nhead,
                    size_t nkvhead,
                    size_t d,
                    size_t dv,
                    size_t total_len,
                    float scale);

} // namespace wginfer::ops::metax


#include "rope_partial_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
void rope_partial_impl(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seqlen,
    size_t nhead,
    size_t head_dim,
    size_t rotary_dim,
    float theta) {
    const size_t half = rotary_dim / 2;
    const size_t total = seqlen * nhead;
    for (size_t idx = 0; idx < total; ++idx) {
        const size_t seq = idx / nhead;
        const size_t base = idx * head_dim;
        const float pos = static_cast<float>(pos_ids[seq]);
        for (size_t j = 0; j < half; ++j) {
            const float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(rotary_dim);
            const float phi = pos / std::pow(theta, exponent);
            const float sinv = std::sin(phi);
            const float cosv = std::cos(phi);
            const float a = wginfer::utils::cast<float>(in[base + j]);
            const float b = wginfer::utils::cast<float>(in[base + j + half]);
            out[base + j] = wginfer::utils::cast<T>(a * cosv - b * sinv);
            out[base + j + half] = wginfer::utils::cast<T>(b * cosv + a * sinv);
        }
        for (size_t j = rotary_dim; j < head_dim; ++j) {
            out[base + j] = in[base + j];
        }
    }
}

} // namespace

namespace wginfer::ops::cpu {

void rope_partial(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t nhead,
    size_t head_dim,
    size_t rotary_dim,
    float theta) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return rope_partial_impl(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos_ids, seqlen, nhead, head_dim, rotary_dim, theta);
    case WGINFER_DTYPE_F16:
        return rope_partial_impl(reinterpret_cast<wginfer::fp16_t *>(out), reinterpret_cast<const wginfer::fp16_t *>(in), pos_ids, seqlen, nhead, head_dim, rotary_dim, theta);
    case WGINFER_DTYPE_BF16:
        return rope_partial_impl(reinterpret_cast<wginfer::bf16_t *>(out), reinterpret_cast<const wginfer::bf16_t *>(in), pos_ids, seqlen, nhead, head_dim, rotary_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::cpu

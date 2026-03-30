#include "causal_conv1d_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void causal_conv1d_(T *out, const T *in, const T *weight, size_t seqlen, size_t channels, size_t kernel_size) {
    for (size_t seq = 0; seq < seqlen; ++seq) {
        for (size_t c = 0; c < channels; ++c) {
            float acc = 0.0f;
            for (size_t k = 0; k < kernel_size; ++k) {
                const size_t input_seq = seq + k;
                if (input_seq >= kernel_size - 1 && (input_seq - (kernel_size - 1)) < seqlen) {
                    const size_t src_seq = input_seq - (kernel_size - 1);
                    const float x = wginfer::utils::cast<float>(in[src_seq * channels + c]);
                    const float w = wginfer::utils::cast<float>(weight[c * kernel_size + k]);
                    acc += x * w;
                } else {
                    continue;
                }
            }
            const float silu = acc / (1.0f + std::exp(-acc));
            out[seq * channels + c] = wginfer::utils::cast<T>(silu);
        }
    }
}

namespace wginfer::ops::cpu {

void causal_conv1d(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t channels,
    size_t kernel_size) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return causal_conv1d_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            seqlen,
            channels,
            kernel_size);
    case WGINFER_DTYPE_F16:
        return causal_conv1d_(
            reinterpret_cast<wginfer::fp16_t *>(out),
            reinterpret_cast<const wginfer::fp16_t *>(in),
            reinterpret_cast<const wginfer::fp16_t *>(weight),
            seqlen,
            channels,
            kernel_size);
    case WGINFER_DTYPE_BF16:
        return causal_conv1d_(
            reinterpret_cast<wginfer::bf16_t *>(out),
            reinterpret_cast<const wginfer::bf16_t *>(in),
            reinterpret_cast<const wginfer::bf16_t *>(weight),
            seqlen,
            channels,
            kernel_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::cpu

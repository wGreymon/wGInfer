#include "gated_rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void gated_rms_norm_(T *out, const T *in, const T *gate, const T *weight, size_t M, size_t N, float eps) {
    for (size_t m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; ++n) {
            const float value = wginfer::utils::cast<float>(in[m * N + n]);
            sum += value * value;
        }
        const float mean = sum / static_cast<float>(N);
        const float scale_rms = 1.0f / std::sqrt(mean + eps);

        for (size_t n = 0; n < N; ++n) {
            const float value = wginfer::utils::cast<float>(in[m * N + n]);
            const float gate_value = wginfer::utils::cast<float>(gate[m * N + n]);
            const float weight_value = wginfer::utils::cast<float>(weight[n]);
            const float silu_gate = gate_value / (1.0f + std::exp(-gate_value));
            const float result = value * scale_rms * weight_value * silu_gate;
            out[m * N + n] = wginfer::utils::cast<T>(result);
        }
    }
}

namespace wginfer::ops::cpu {

void gated_rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *gate,
    const std::byte *weight,
    wginferDataType_t dtype,
    size_t M,
    size_t N,
    float eps) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return gated_rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(weight),
            M,
            N,
            eps);
    case WGINFER_DTYPE_F16:
        return gated_rms_norm_(
            reinterpret_cast<wginfer::fp16_t *>(out),
            reinterpret_cast<const wginfer::fp16_t *>(in),
            reinterpret_cast<const wginfer::fp16_t *>(gate),
            reinterpret_cast<const wginfer::fp16_t *>(weight),
            M,
            N,
            eps);
    case WGINFER_DTYPE_BF16:
        return gated_rms_norm_(
            reinterpret_cast<wginfer::bf16_t *>(out),
            reinterpret_cast<const wginfer::bf16_t *>(in),
            reinterpret_cast<const wginfer::bf16_t *>(gate),
            reinterpret_cast<const wginfer::bf16_t *>(weight),
            M,
            N,
            eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::cpu

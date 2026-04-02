#include "normalize_l2_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace wginfer::ops::cpu {

namespace {
template <typename T>
void normalize_l2_kernel(T *out, const T *in, size_t M, size_t N, float eps, float scale) {
    for (size_t i = 0; i < M; ++i) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            float v = wginfer::utils::cast<float>(in[i * N + j]);
            sum_sq += v * v;
        }
        float inv_norm = scale / std::sqrt(sum_sq + eps);
        for (size_t j = 0; j < N; ++j) {
            float v = wginfer::utils::cast<float>(in[i * N + j]);
            out[i * N + j] = wginfer::utils::cast<T>(v * inv_norm);
        }
    }
}
} // namespace

void normalize_l2(
    std::byte *out,
    const std::byte *in,
    wginferDataType_t dtype,
    size_t M,
    size_t N,
    float eps,
    float scale) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        normalize_l2_kernel(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            M, N, eps, scale);
        break;
    case WGINFER_DTYPE_F16:
        normalize_l2_kernel(
            reinterpret_cast<wginfer::fp16_t *>(out),
            reinterpret_cast<const wginfer::fp16_t *>(in),
            M, N, eps, scale);
        break;
    case WGINFER_DTYPE_BF16:
        normalize_l2_kernel(
            reinterpret_cast<wginfer::bf16_t *>(out),
            reinterpret_cast<const wginfer::bf16_t *>(in),
            M, N, eps, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::cpu

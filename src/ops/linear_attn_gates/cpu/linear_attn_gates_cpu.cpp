#include "linear_attn_gates_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {

inline float softplus_stable(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return std::exp(x);
    }
    return std::log1p(std::exp(x));
}

template <typename T>
void linear_attn_gates_impl(T *out_g, T *out_beta, const T *a, const T *b, const T *a_log, const T *dt_bias, size_t M, size_t H) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t h = 0; h < H; ++h) {
            const size_t idx = i * H + h;
            const float a_val = wginfer::utils::cast<float>(a[idx]);
            const float b_val = wginfer::utils::cast<float>(b[idx]);
            const float a_log_val = wginfer::utils::cast<float>(a_log[h]);
            const float dt_val = wginfer::utils::cast<float>(dt_bias[h]);
            out_g[idx] = wginfer::utils::cast<T>(-std::exp(a_log_val) * softplus_stable(a_val + dt_val));
            out_beta[idx] = wginfer::utils::cast<T>(1.0f / (1.0f + std::exp(-b_val)));
        }
    }
}

} // namespace

namespace wginfer::ops::cpu {

void linear_attn_gates(std::byte *out_g, std::byte *out_beta, const std::byte *a, const std::byte *b, const std::byte *a_log, const std::byte *dt_bias, wginferDataType_t dtype, size_t M, size_t H) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return linear_attn_gates_impl(reinterpret_cast<float *>(out_g), reinterpret_cast<float *>(out_beta), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), reinterpret_cast<const float *>(a_log), reinterpret_cast<const float *>(dt_bias), M, H);
    case WGINFER_DTYPE_F16:
        return linear_attn_gates_impl(reinterpret_cast<wginfer::fp16_t *>(out_g), reinterpret_cast<wginfer::fp16_t *>(out_beta), reinterpret_cast<const wginfer::fp16_t *>(a), reinterpret_cast<const wginfer::fp16_t *>(b), reinterpret_cast<const wginfer::fp16_t *>(a_log), reinterpret_cast<const wginfer::fp16_t *>(dt_bias), M, H);
    case WGINFER_DTYPE_BF16:
        return linear_attn_gates_impl(reinterpret_cast<wginfer::bf16_t *>(out_g), reinterpret_cast<wginfer::bf16_t *>(out_beta), reinterpret_cast<const wginfer::bf16_t *>(a), reinterpret_cast<const wginfer::bf16_t *>(b), reinterpret_cast<const wginfer::bf16_t *>(a_log), reinterpret_cast<const wginfer::bf16_t *>(dt_bias), M, H);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace wginfer::ops::cpu

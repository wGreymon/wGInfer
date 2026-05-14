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

float read_value(const std::byte *data, wginferDataType_t dtype, size_t idx) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return reinterpret_cast<const float *>(data)[idx];
    case WGINFER_DTYPE_F16:
        return wginfer::utils::cast<float>(reinterpret_cast<const wginfer::fp16_t *>(data)[idx]);
    case WGINFER_DTYPE_BF16:
        return wginfer::utils::cast<float>(reinterpret_cast<const wginfer::bf16_t *>(data)[idx]);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

template <typename OutT, typename InT>
void linear_attn_gates_impl(
    OutT *out_g,
    OutT *out_beta,
    const InT *a,
    const InT *b,
    const std::byte *a_log,
    const std::byte *dt_bias,
    wginferDataType_t a_log_dtype,
    wginferDataType_t dt_bias_dtype,
    size_t M,
    size_t H) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t h = 0; h < H; ++h) {
            const size_t idx = i * H + h;
            const float a_val = wginfer::utils::cast<float>(a[idx]);
            const float b_val = wginfer::utils::cast<float>(b[idx]);
            const float a_log_val = read_value(a_log, a_log_dtype, h);
            const float dt_val = read_value(dt_bias, dt_bias_dtype, h);
            out_g[idx] = wginfer::utils::cast<OutT>(-std::exp(a_log_val) * softplus_stable(a_val + dt_val));
            out_beta[idx] = wginfer::utils::cast<OutT>(1.0f / (1.0f + std::exp(-b_val)));
        }
    }
}

template <typename OutT>
void dispatch_input_dtype(
    std::byte *out_g,
    std::byte *out_beta,
    const std::byte *a,
    const std::byte *b,
    const std::byte *a_log,
    const std::byte *dt_bias,
    wginferDataType_t input_dtype,
    wginferDataType_t a_log_dtype,
    wginferDataType_t dt_bias_dtype,
    size_t M,
    size_t H) {
    switch (input_dtype) {
    case WGINFER_DTYPE_F32:
        return linear_attn_gates_impl(reinterpret_cast<OutT *>(out_g), reinterpret_cast<OutT *>(out_beta), reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b), a_log, dt_bias, a_log_dtype, dt_bias_dtype, M, H);
    case WGINFER_DTYPE_F16:
        return linear_attn_gates_impl(reinterpret_cast<OutT *>(out_g), reinterpret_cast<OutT *>(out_beta), reinterpret_cast<const wginfer::fp16_t *>(a), reinterpret_cast<const wginfer::fp16_t *>(b), a_log, dt_bias, a_log_dtype, dt_bias_dtype, M, H);
    case WGINFER_DTYPE_BF16:
        return linear_attn_gates_impl(reinterpret_cast<OutT *>(out_g), reinterpret_cast<OutT *>(out_beta), reinterpret_cast<const wginfer::bf16_t *>(a), reinterpret_cast<const wginfer::bf16_t *>(b), a_log, dt_bias, a_log_dtype, dt_bias_dtype, M, H);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(input_dtype);
    }
}

} // namespace

namespace wginfer::ops::cpu {

void linear_attn_gates(
    std::byte *out_g,
    std::byte *out_beta,
    const std::byte *a,
    const std::byte *b,
    const std::byte *a_log,
    const std::byte *dt_bias,
    wginferDataType_t out_dtype,
    wginferDataType_t input_dtype,
    wginferDataType_t a_log_dtype,
    wginferDataType_t dt_bias_dtype,
    size_t M,
    size_t H) {
    switch (out_dtype) {
    case WGINFER_DTYPE_F32:
        return dispatch_input_dtype<float>(out_g, out_beta, a, b, a_log, dt_bias, input_dtype, a_log_dtype, dt_bias_dtype, M, H);
    case WGINFER_DTYPE_F16:
        return dispatch_input_dtype<wginfer::fp16_t>(out_g, out_beta, a, b, a_log, dt_bias, input_dtype, a_log_dtype, dt_bias_dtype, M, H);
    case WGINFER_DTYPE_BF16:
        return dispatch_input_dtype<wginfer::bf16_t>(out_g, out_beta, a, b, a_log, dt_bias, input_dtype, a_log_dtype, dt_bias_dtype, M, H);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out_dtype);
    }
}

} // namespace wginfer::ops::cpu

#include "linear_attn_gates_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

namespace {

__device__ __forceinline__ float softplus_stable(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

__device__ __forceinline__ float read_param(const std::byte *data, wginferDataType_t dtype, size_t idx) {
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        return reinterpret_cast<const float *>(data)[idx];
    case WGINFER_DTYPE_F16:
        return __half2float(reinterpret_cast<const half *>(data)[idx]);
    case WGINFER_DTYPE_BF16:
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(data)[idx]);
    default:
        return 0.0f;
    }
}

template <typename OutT, typename InT>
__global__ void linear_attn_gates_kernel(
    OutT *out_g,
    OutT *out_beta,
    const InT *a,
    const InT *b,
    const std::byte *a_log,
    const std::byte *dt_bias,
    wginferDataType_t a_log_dtype,
    wginferDataType_t dt_bias_dtype,
    size_t total,
    size_t H) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const size_t h = idx % H;
    const float a_val = to_float(a[idx]);
    const float b_val = to_float(b[idx]);
    const float a_log_val = read_param(a_log, a_log_dtype, h);
    const float dt_val = read_param(dt_bias, dt_bias_dtype, h);
    out_g[idx] = from_float<OutT>(-expf(a_log_val) * softplus_stable(a_val + dt_val));
    out_beta[idx] = from_float<OutT>(1.0f / (1.0f + expf(-b_val)));
}

template <typename OutT, typename InT>
void launch_linear_attn_gates(
    std::byte *out_g,
    std::byte *out_beta,
    const std::byte *a,
    const std::byte *b,
    const std::byte *a_log,
    const std::byte *dt_bias,
    wginferDataType_t a_log_dtype,
    wginferDataType_t dt_bias_dtype,
    size_t M,
    size_t H) {
    const size_t total = M * H;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(CEIL(total, static_cast<size_t>(block_size)));
    linear_attn_gates_kernel<OutT, InT><<<grid_size, block_size>>>(
        reinterpret_cast<OutT *>(out_g),
        reinterpret_cast<OutT *>(out_beta),
        reinterpret_cast<const InT *>(a),
        reinterpret_cast<const InT *>(b),
        a_log,
        dt_bias,
        a_log_dtype,
        dt_bias_dtype,
        total,
        H);
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
        return launch_linear_attn_gates<OutT, float>(out_g, out_beta, a, b, a_log, dt_bias, a_log_dtype, dt_bias_dtype, M, H);
    case WGINFER_DTYPE_F16:
        return launch_linear_attn_gates<OutT, half>(out_g, out_beta, a, b, a_log, dt_bias, a_log_dtype, dt_bias_dtype, M, H);
    case WGINFER_DTYPE_BF16:
        return launch_linear_attn_gates<OutT, __nv_bfloat16>(out_g, out_beta, a, b, a_log, dt_bias, a_log_dtype, dt_bias_dtype, M, H);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(input_dtype);
    }
}

} // namespace

namespace wginfer::ops::nvidia {

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
    if (M == 0 || H == 0) {
        return;
    }
    switch (out_dtype) {
    case WGINFER_DTYPE_F32:
        dispatch_input_dtype<float>(out_g, out_beta, a, b, a_log, dt_bias, input_dtype, a_log_dtype, dt_bias_dtype, M, H);
        break;
    case WGINFER_DTYPE_F16:
        dispatch_input_dtype<half>(out_g, out_beta, a, b, a_log, dt_bias, input_dtype, a_log_dtype, dt_bias_dtype, M, H);
        break;
    case WGINFER_DTYPE_BF16:
        dispatch_input_dtype<__nv_bfloat16>(out_g, out_beta, a, b, a_log, dt_bias, input_dtype, a_log_dtype, dt_bias_dtype, M, H);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out_dtype);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia

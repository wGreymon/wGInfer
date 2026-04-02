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

template <typename T>
__global__ void linear_attn_gates_kernel(T *out_g, T *out_beta, const T *a, const T *b, const T *a_log, const T *dt_bias, size_t total, size_t H) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const size_t h = idx % H;
    const float a_val = to_float(a[idx]);
    const float b_val = to_float(b[idx]);
    const float a_log_val = to_float(a_log[h]);
    const float dt_val = to_float(dt_bias[h]);
    out_g[idx] = from_float<T>(-expf(a_log_val) * softplus_stable(a_val + dt_val));
    out_beta[idx] = from_float<T>(1.0f / (1.0f + expf(-b_val)));
}

template <typename T>
void launch_linear_attn_gates(std::byte *out_g, std::byte *out_beta, const std::byte *a, const std::byte *b, const std::byte *a_log, const std::byte *dt_bias, size_t M, size_t H) {
    const size_t total = M * H;
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(CEIL(total, static_cast<size_t>(block_size)));
    linear_attn_gates_kernel<T><<<grid_size, block_size>>>(reinterpret_cast<T *>(out_g), reinterpret_cast<T *>(out_beta), reinterpret_cast<const T *>(a), reinterpret_cast<const T *>(b), reinterpret_cast<const T *>(a_log), reinterpret_cast<const T *>(dt_bias), total, H);
}

} // namespace

namespace wginfer::ops::nvidia {

void linear_attn_gates(std::byte *out_g, std::byte *out_beta, const std::byte *a, const std::byte *b, const std::byte *a_log, const std::byte *dt_bias, wginferDataType_t dtype, size_t M, size_t H) {
    if (M == 0 || H == 0) {
        return;
    }
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        launch_linear_attn_gates<float>(out_g, out_beta, a, b, a_log, dt_bias, M, H);
        break;
    case WGINFER_DTYPE_F16:
        launch_linear_attn_gates<half>(out_g, out_beta, a, b, a_log, dt_bias, M, H);
        break;
    case WGINFER_DTYPE_BF16:
        launch_linear_attn_gates<__nv_bfloat16>(out_g, out_beta, a, b, a_log, dt_bias, M, H);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia

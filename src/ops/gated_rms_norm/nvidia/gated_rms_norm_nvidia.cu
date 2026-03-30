#include "gated_rms_norm_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

#include <cstddef>

namespace {

template <typename T>
__device__ __forceinline__ float warp_reduce_sum_float(float local_val) {
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        local_val += __shfl_xor_sync(0xffffffff, local_val, stride);
    }
    return local_val;
}

template <const int BLOCK_SIZE = 256>
__device__ __forceinline__ float block_reduce_sum_float(float local_val) {
    constexpr int warp_per_block = CEIL(BLOCK_SIZE, 32);
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    __shared__ float shared_val[warp_per_block];

    local_val = warp_reduce_sum_float<float>(local_val);
    if (lane_id == 0) {
        shared_val[warp_id] = local_val;
    }
    __syncthreads();

    float lane_val = lane_id < warp_per_block ? shared_val[lane_id] : 0.0f;
    return warp_reduce_sum_float<float>(lane_val);
}

template <typename T>
__global__ void gated_rms_norm_kernel(
    T *out,
    const T *in,
    const T *gate,
    const T *weight,
    size_t M,
    size_t N,
    float eps) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= M) {
        return;
    }

    float sum_thread = 0.0f;
    for (size_t col = static_cast<size_t>(threadIdx.x); col < N; col += static_cast<size_t>(blockDim.x)) {
        const float x = to_float(in[row * N + col]);
        sum_thread += x * x;
    }

    const float sum_block = block_reduce_sum_float<>(sum_thread);
    const float mean_sq = sum_block / static_cast<float>(N);
    const float scale_rms = rsqrtf(mean_sq + eps);

    for (size_t col = static_cast<size_t>(threadIdx.x); col < N; col += static_cast<size_t>(blockDim.x)) {
        const float x = to_float(in[row * N + col]);
        const float g = to_float(gate[row * N + col]);
        const float w = to_float(weight[col]);
        const float silu = g / (1.0f + expf(-g));
        const float y = x * scale_rms * w * silu;
        out[row * N + col] = from_float<T>(y);
    }
}

template <typename T>
void launch_gated_rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *gate,
    const std::byte *weight,
    size_t M,
    size_t N,
    float eps) {
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(M);
    gated_rms_norm_kernel<T><<<grid_size, block_size>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(weight),
        M,
        N,
        eps);
}

} // namespace

namespace wginfer::ops::nvidia {

void gated_rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *gate,
    const std::byte *weight,
    wginferDataType_t dtype,
    size_t M,
    size_t N,
    float eps) {
    if (M == 0 || N == 0) {
        return;
    }

    switch (dtype) {
    case WGINFER_DTYPE_F32:
        launch_gated_rms_norm<float>(out, in, gate, weight, M, N, eps);
        break;
    case WGINFER_DTYPE_F16:
        launch_gated_rms_norm<half>(out, in, gate, weight, M, N, eps);
        break;
    case WGINFER_DTYPE_BF16:
        launch_gated_rms_norm<__nv_bfloat16>(out, in, gate, weight, M, N, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia

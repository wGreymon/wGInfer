#include "normalize_l2_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

namespace {

template <const int BLOCK_SIZE = 256>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int warps = (BLOCK_SIZE + 31) / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    __shared__ float smem[warps];

#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    float lane_val = lane_id < warps ? smem[lane_id] : 0.0f;
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        lane_val += __shfl_xor_sync(0xffffffff, lane_val, stride);
    }
    return lane_val;
}

template <typename T>
__global__ void normalize_l2_kernel(T *out, const T *in, size_t M, size_t N, float eps, float scale) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= M) {
        return;
    }

    float sum_sq = 0.0f;
    for (size_t j = static_cast<size_t>(threadIdx.x); j < N; j += static_cast<size_t>(blockDim.x)) {
        float v = to_float(in[row * N + j]);
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum<256>(sum_sq);

    const float inv_norm = scale / sqrtf(sum_sq + eps);
    for (size_t j = static_cast<size_t>(threadIdx.x); j < N; j += static_cast<size_t>(blockDim.x)) {
        float v = to_float(in[row * N + j]);
        out[row * N + j] = from_float<T>(v * inv_norm);
    }
}

} // namespace

namespace wginfer::ops::nvidia {

void normalize_l2(
    std::byte *out,
    const std::byte *in,
    wginferDataType_t dtype,
    size_t M,
    size_t N,
    float eps,
    float scale) {
    if (M == 0 || N == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(M);
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        normalize_l2_kernel<float><<<grid_size, block_size>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), M, N, eps, scale);
        break;
    case WGINFER_DTYPE_F16:
        normalize_l2_kernel<half><<<grid_size, block_size>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in), M, N, eps, scale);
        break;
    case WGINFER_DTYPE_BF16:
        normalize_l2_kernel<__nv_bfloat16><<<grid_size, block_size>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in), M, N, eps, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia

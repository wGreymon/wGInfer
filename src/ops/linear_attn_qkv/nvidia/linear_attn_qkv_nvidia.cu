#include "linear_attn_qkv_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

namespace {

template <const int BLOCK_SIZE = 256>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int warps = (BLOCK_SIZE + 31) / 32;
    const int lane_id = threadIdx.x % 32;
    __shared__ float smem[warps];

#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    if (lane_id == 0) {
        smem[threadIdx.x / 32] = val;
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
__global__ void prepare_qk_kernel(
    T *out_q,
    T *out_k,
    const T *mixed_qkv,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    float eps,
    float q_scale) {
    const size_t row_idx = static_cast<size_t>(blockIdx.x);
    const size_t seq = row_idx / key_heads;
    const size_t key_head = row_idx % key_heads;
    const size_t repeat_factor = value_heads / key_heads;
    const size_t key_total_dim = key_heads * key_dim;
    const size_t qkv_dim = 2 * key_total_dim + value_heads * value_dim;
    const T *row = mixed_qkv + seq * qkv_dim;
    const T *q_src = row + key_head * key_dim;
    const T *k_src = row + key_total_dim + key_head * key_dim;

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (size_t d = static_cast<size_t>(threadIdx.x); d < key_dim; d += static_cast<size_t>(blockDim.x)) {
        const float qv = to_float(q_src[d]);
        const float kv = to_float(k_src[d]);
        q_sum_sq += qv * qv;
        k_sum_sq += kv * kv;
    }

    q_sum_sq = block_reduce_sum<256>(q_sum_sq);
    __syncthreads();
    k_sum_sq = block_reduce_sum<256>(k_sum_sq);
    const float q_inv_norm = q_scale * rsqrtf(q_sum_sq + eps);
    const float k_inv_norm = rsqrtf(k_sum_sq + eps);

    for (size_t d = static_cast<size_t>(threadIdx.x); d < key_dim; d += static_cast<size_t>(blockDim.x)) {
        const float qv = to_float(q_src[d]) * q_inv_norm;
        const float kv = to_float(k_src[d]) * k_inv_norm;
        for (size_t rep = 0; rep < repeat_factor; ++rep) {
            const size_t value_head = key_head * repeat_factor + rep;
            const size_t dst_base = (seq * value_heads + value_head) * key_dim + d;
            out_q[dst_base] = from_float<T>(qv);
            out_k[dst_base] = from_float<T>(kv);
        }
    }
}

template <typename T>
__global__ void prepare_v_kernel(
    T *out_v,
    const T *mixed_qkv,
    size_t total,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + static_cast<size_t>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const size_t d = idx % value_dim;
    const size_t value_head = (idx / value_dim) % value_heads;
    const size_t seq = idx / (value_heads * value_dim);
    const size_t key_total_dim = key_heads * key_dim;
    const size_t qkv_dim = 2 * key_total_dim + value_heads * value_dim;
    const size_t src_idx = seq * qkv_dim + 2 * key_total_dim + value_head * value_dim + d;
    out_v[idx] = mixed_qkv[src_idx];
}

template <typename T>
void launch_linear_attn_qkv_prepare(
    std::byte *out_q,
    std::byte *out_k,
    std::byte *out_v,
    const std::byte *mixed_qkv,
    size_t seqlen,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    float eps,
    float q_scale) {
    constexpr int block_size = 256;
    const int qk_grid_size = static_cast<int>(seqlen * key_heads);
    prepare_qk_kernel<T><<<qk_grid_size, block_size>>>(
        reinterpret_cast<T *>(out_q),
        reinterpret_cast<T *>(out_k),
        reinterpret_cast<const T *>(mixed_qkv),
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        eps,
        q_scale);

    const size_t v_total = seqlen * value_heads * value_dim;
    const int v_grid_size = static_cast<int>(CEIL(v_total, static_cast<size_t>(block_size)));
    prepare_v_kernel<T><<<v_grid_size, block_size>>>(
        reinterpret_cast<T *>(out_v),
        reinterpret_cast<const T *>(mixed_qkv),
        v_total,
        key_heads,
        value_heads,
        key_dim,
        value_dim);
}

} // namespace

namespace wginfer::ops::nvidia {

void linear_attn_qkv_prepare(
    std::byte *out_q,
    std::byte *out_k,
    std::byte *out_v,
    const std::byte *mixed_qkv,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t key_heads,
    size_t value_heads,
    size_t key_dim,
    size_t value_dim,
    float eps,
    float q_scale) {
    if (seqlen == 0 || key_heads == 0 || value_heads == 0 || key_dim == 0 || value_dim == 0) {
        return;
    }

    switch (dtype) {
    case WGINFER_DTYPE_F32:
        launch_linear_attn_qkv_prepare<float>(
            out_q, out_k, out_v, mixed_qkv, seqlen, key_heads, value_heads, key_dim, value_dim, eps, q_scale);
        break;
    case WGINFER_DTYPE_F16:
        launch_linear_attn_qkv_prepare<half>(
            out_q, out_k, out_v, mixed_qkv, seqlen, key_heads, value_heads, key_dim, value_dim, eps, q_scale);
        break;
    case WGINFER_DTYPE_BF16:
        launch_linear_attn_qkv_prepare<__nv_bfloat16>(
            out_q, out_k, out_v, mixed_qkv, seqlen, key_heads, value_heads, key_dim, value_dim, eps, q_scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia

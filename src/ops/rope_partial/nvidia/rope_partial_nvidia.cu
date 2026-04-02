#include "rope_partial_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/gpu_utils.hpp"

namespace {

template <typename T>
__global__ void rope_partial_kernel(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seqlen,
    size_t nhead,
    size_t head_dim,
    size_t rotary_dim,
    float theta) {
    const size_t bid = static_cast<size_t>(blockIdx.x);
    if (bid >= seqlen * nhead) {
        return;
    }

    const size_t seq = bid / nhead;
    const size_t base = bid * head_dim;
    const size_t half = rotary_dim / 2;
    const float pos = static_cast<float>(pos_ids[seq]);

    for (size_t j = static_cast<size_t>(threadIdx.x); j < half; j += static_cast<size_t>(blockDim.x)) {
        const float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(rotary_dim);
        const float phi = pos / powf(theta, exponent);
        const float sinv = sinf(phi);
        const float cosv = cosf(phi);
        const float a = to_float(in[base + j]);
        const float b = to_float(in[base + j + half]);
        out[base + j] = from_float<T>(a * cosv - b * sinv);
        out[base + j + half] = from_float<T>(b * cosv + a * sinv);
    }
    for (size_t j = rotary_dim + static_cast<size_t>(threadIdx.x); j < head_dim; j += static_cast<size_t>(blockDim.x)) {
        out[base + j] = in[base + j];
    }
}

} // namespace

namespace wginfer::ops::nvidia {

void rope_partial(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    wginferDataType_t dtype,
    size_t seqlen,
    size_t nhead,
    size_t head_dim,
    size_t rotary_dim,
    float theta) {
    if (seqlen == 0 || nhead == 0 || head_dim == 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(seqlen * nhead);
    switch (dtype) {
    case WGINFER_DTYPE_F32:
        rope_partial_kernel<float><<<grid_size, block_size>>>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos_ids, seqlen, nhead, head_dim, rotary_dim, theta);
        break;
    case WGINFER_DTYPE_F16:
        rope_partial_kernel<half><<<grid_size, block_size>>>(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in), pos_ids, seqlen, nhead, head_dim, rotary_dim, theta);
        break;
    case WGINFER_DTYPE_BF16:
        rope_partial_kernel<__nv_bfloat16><<<grid_size, block_size>>>(reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in), pos_ids, seqlen, nhead, head_dim, rotary_dim, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace wginfer::ops::nvidia
